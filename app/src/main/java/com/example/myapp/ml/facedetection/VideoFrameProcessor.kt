package com.example.myapp.ml.facedetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.Rect
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.max
import kotlin.math.min

/**
 * One decoded video frame, together with every face ML Kit detected in it.
 */
data class FrameData(
    val timestampMs: Long,
    val frameBitmap: Bitmap,
    val faces: List<DetectedFaceInstance>
)

/**
 * A single detected face within a specific video frame.
 */
data class DetectedFaceInstance(
    val timestampMs: Long,
    val fullFrameBitmap: Bitmap,
    val faceBitmap: Bitmap,
    val boundingBox: Rect,
    val headEulerAngleY: Float,
    val headEulerAngleZ: Float,
    val leftEyeOpenProbability: Float?,
    val rightEyeOpenProbability: Float?,
    val smilingProbability: Float?,
    val sharpnessScore: Double,
    /**
     * Eye positions in [fullFrameBitmap] coordinate space (subject's own left/right, i.e. as
     * ML Kit reports them — mirrored relative to what a viewer sees), when ML Kit was able to
     * find them. Used by [com.example.myapp.ml.embedding.FaceEmbeddingExtractor.alignFace] to
     * rotate/scale each crop to a canonical pose before embedding, which matters a lot for
     * embedding quality/consistency — MobileFaceNet-family models are trained on eye-aligned
     * crops and are noticeably less discriminative on raw, un-aligned bounding-box crops where
     * head roll varies frame to frame. Null when either eye landmark wasn't detected (profile
     * views, occlusion, etc.) — callers should fall back to a plain padded crop in that case.
     */
    val leftEyePosition: PointF?,
    val rightEyePosition: PointF?,
    val otherFaceBoxesInFrame: List<Rect>
)

/**
 * Extracts frames from a local/content video URI at a fixed interval, runs ML Kit face
 * detection on each frame, and attaches a no-dependency sharpness estimate (variance of a
 * discrete Laplacian) to every detected face. All decoding/inference work runs on
 * [Dispatchers.IO].
 *
 * Usage:
 * ```
 * val processor = VideoFrameProcessor()
 * processor.extractFrames(context, videoUri)
 *     .collect { frame -> /* frame.faces: List<DetectedFaceInstance> */ }
 * ```
 */
class VideoFrameProcessor {

    private val faceDetectorOptions = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
        // ALL (not NONE) so we get eye-position landmarks — needed to rotate/scale each face
        // crop to a canonical pose before embedding. Un-aligned crops are the single biggest
        // hit to MobileFaceNet-family embedding quality/consistency, worse than any clustering
        // threshold tweak can compensate for.
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
        // Ignore faces smaller than 8% of the frame width: at that size ML Kit's own detections
        // get noisy (motion-blur smears, distant background extras) and the resulting crops are
        // too low-resolution to embed reliably — they mostly just add junk clusters.
        .setMinFaceSize(0.08f)
        .build()

    /**
     * Emits one [FrameData] per sampled frame, in timestamp order, from 0ms to the video's
     * duration, stepping by [frameIntervalMs]. Frames that fail to decode are silently skipped;
     * frames with zero detected faces are still emitted (with an empty [FrameData.faces] list)
     * so callers can track coverage/progress.
     */
    fun extractFrames(
        context: Context,
        videoUri: Uri,
        frameIntervalMs: Long = 150
    ): Flow<FrameData> = flow {
        val retriever = MediaMetadataRetriever()
        val detector = FaceDetection.getClient(faceDetectorOptions)

        try {
            retriever.setDataSource(context, videoUri)

            val durationMs = retriever
                .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: 0L

            require(frameIntervalMs > 0) { "frameIntervalMs must be positive" }

            var currentMs = 0L
            while (currentMs <= durationMs) {
                val frameBitmap = runCatching {
                    retriever.getFrameAtTime(
                        currentMs * 1000, // retriever expects microseconds
                        MediaMetadataRetriever.OPTION_CLOSEST
                    )
                }.getOrNull()

                if (frameBitmap != null) {
                    val faces = detectFacesInFrame(detector, frameBitmap, currentMs)
                    emit(FrameData(currentMs, frameBitmap, faces))
                }

                currentMs += frameIntervalMs
            }
        } finally {
            retriever.release()
            detector.close()
        }
    }.flowOn(Dispatchers.IO)

    private suspend fun detectFacesInFrame(
        detector: FaceDetector,
        frameBitmap: Bitmap,
        timestampMs: Long
    ): List<DetectedFaceInstance> {
        val inputImage = InputImage.fromBitmap(frameBitmap, 0)
        val faces: List<Face> = detector.process(inputImage).await()

        val safeBoxes = faces.map { clampBoundingBox(it.boundingBox, frameBitmap.width, frameBitmap.height) }

        return faces.mapIndexedNotNull { index, face ->
            val safeBox = safeBoxes[index]
            if (safeBox.width() <= 0 || safeBox.height() <= 0) return@mapIndexedNotNull null

            val faceBitmap = Bitmap.createBitmap(frameBitmap, safeBox.left, safeBox.top, safeBox.width(), safeBox.height())
            val otherBoxes = safeBoxes.filterIndexed { i, _ -> i != index }

            DetectedFaceInstance(
                timestampMs = timestampMs,
                fullFrameBitmap = frameBitmap,
                faceBitmap = faceBitmap,
                boundingBox = face.boundingBox,
                headEulerAngleY = face.headEulerAngleY,
                headEulerAngleZ = face.headEulerAngleZ,
                leftEyeOpenProbability = face.leftEyeOpenProbability,
                rightEyeOpenProbability = face.rightEyeOpenProbability,
                smilingProbability = face.smilingProbability,
                sharpnessScore = laplacianVarianceSharpness(faceBitmap),
                leftEyePosition = face.getLandmark(FaceLandmark.LEFT_EYE)?.position,
                rightEyePosition = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position,
                otherFaceBoxesInFrame = otherBoxes
            )
        }
    }

    private fun nearestNeighborGap(box: Rect, allBoxes: List<Rect>, selfIndex: Int): Float? {
        var minGap: Float? = null
        allBoxes.forEachIndexed { i, other ->
            if (i == selfIndex) return@forEachIndexed
            val verticalOverlap = min(box.bottom, other.bottom) - max(box.top, other.top)
            if (verticalOverlap <= 0) return@forEachIndexed

            val gap = when {
                other.left >= box.right -> (other.left - box.right).toFloat()
                box.left >= other.right -> (box.left - other.right).toFloat()
                else -> 0f // boxes already overlap
            }
            if (minGap == null || gap < minGap!!) minGap = gap
        }
        return minGap
    }

    /** ML Kit's bounding box can extend outside the bitmap; clip it before cropping. */
    private fun clampBoundingBox(box: Rect, bitmapWidth: Int, bitmapHeight: Int): Rect {
        val left = max(0, box.left)
        val top = max(0, box.top)
        val right = min(bitmapWidth, box.right)
        val bottom = min(bitmapHeight, box.bottom)
        return Rect(left, top, right, bottom)
    }



    /**
     * Computes a blur/sharpness estimate with no OpenCV dependency: converts the bitmap to
     * grayscale, convolves with a discrete 3x3 Laplacian kernel, and returns the variance of
     * the resulting edge-response values. Higher variance => more high-frequency detail =>
     * sharper image. This mirrors the common "variance of Laplacian" blur-detection heuristic.
     */
    private fun laplacianVarianceSharpness(bitmap: Bitmap): Double {
        val width = bitmap.width
        val height = bitmap.height
        if (width < 3 || height < 3) return 0.0

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // Standard luminance conversion (Rec. 601 coefficients).
        val gray = DoubleArray(width * height)
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            gray[i] = 0.299 * r + 0.587 * g + 0.114 * b
        }

        fun at(x: Int, y: Int) = gray[y * width + x]

        // 3x3 Laplacian kernel: [[0,1,0],[1,-4,1],[0,1,0]], applied to interior pixels only.
        val responses = ArrayList<Double>((width - 2) * (height - 2))
        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                val laplacian = at(x, y - 1) + at(x, y + 1) + at(x - 1, y) + at(x + 1, y) - 4 * at(x, y)
                responses.add(laplacian)
            }
        }

        if (responses.isEmpty()) return 0.0

        val mean = responses.sum() / responses.size
        val variance = responses.sumOf {
            val d = it - mean
            d * d
        } / responses.size

// Normalize sharpness to 0..1.
// Higher raw Laplacian variance = sharper image.
        return variance / (variance + 100.0)
    }

    /** Bridges a Play Services [Task] into a suspend function without pulling in an extra artifact. */
    private suspend fun <T> Task<T>.await(): T = suspendCancellableCoroutine { cont ->
        addOnSuccessListener { result -> cont.resume(result) }
        addOnFailureListener { exception -> cont.resumeWithException(exception) }
        addOnCanceledListener { cont.cancel() }
    }

    private companion object {
        private const val TAG = "VideoFrameProcessor"
    }
}
