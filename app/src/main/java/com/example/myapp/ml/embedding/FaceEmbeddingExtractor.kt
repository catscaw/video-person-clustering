package com.example.myapp.ml.embedding

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.Rect
import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import com.example.myapp.ml.geometry.shrinkToAvoidOverlap

/**
 * Converts a face bitmap into a fixed-length embedding vector using a TensorFlow Lite
 * face-recognition model (e.g. MobileFaceNet, FaceNet) loaded from `assets/`.
 *
 * The model is expected to take a single [inputSize] x [inputSize] x 3 float image and
 * produce a single 1 x [embeddingSize] float output. MobileFaceNet variants commonly use
 * 112x112 input with a 128 or 192-dim output; FaceNet (Inception-ResNet based) variants
 * commonly use 160x160 input with a 128 or 512-dim output. Set the constructor parameters
 * to match whichever `.tflite` file you actually ship.
 *
 * Usage:
 * ```
 * val extractor = FaceEmbeddingExtractor(context, modelFileName = "mobilefacenet.tflite")
 * val paddedCrop = extractor.cropWithPadding(sourceBitmap, face.boundingBox)
 * val embedding = extractor.generateEmbedding(paddedCrop)
 * extractor.close()
 * ```
 */
class FaceEmbeddingExtractor(
    context: Context,
    modelFileName: String = "mobilefacenet.tflite",
    private val inputSize: Int = 112,
    private val embeddingSize: Int = 192,
    numThreads: Int = 4,
    useNnApi: Boolean = false
) {

    private val interpreter: Interpreter

    init {
        val options = Interpreter.Options().apply {
            setNumThreads(numThreads)
            setUseNNAPI(useNnApi)
        }
        interpreter = Interpreter(loadModelFile(context, modelFileName), options)
    }

    private fun loadModelFile(context: Context, fileName: String): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd(fileName)
        FileInputStream(assetFileDescriptor.fileDescriptor).use { inputStream ->
            val fileChannel: FileChannel = inputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }
    }

    /**
     * Expands an ML Kit face [boundingBox] by [paddingFraction] on every side (25% by default)
     * so the crop includes ears/chin/forehead instead of clipping tightly to detected
     * landmarks, then crops it out of [source]. The expanded box is clamped to the bitmap's
     * actual bounds.
     */
    fun cropWithPadding(
        source: Bitmap,
        boundingBox: Rect,
        paddingFraction: Float = 0.25f,
        otherFaceBoxesInFrame: List<Rect> = emptyList()
    ): Bitmap {
        val padX = (boundingBox.width() * paddingFraction).toInt()
        val padY = (boundingBox.height() * paddingFraction).toInt()

        var rect = Rect(
            boundingBox.left - padX,
            boundingBox.top - padY,
            boundingBox.right + padX,
            boundingBox.bottom + padY
        )
           // move/share this helper

        val left = max(0, rect.left)
        val top = max(0, rect.top)
        val right = min(source.width, rect.right)
        val bottom = min(source.height, rect.bottom)

        require(right - left > 0 && bottom - top > 0) {
            "Padded bounding box has non-positive size after clamping"
        }
        return Bitmap.createBitmap(source, left, top, right - left, bottom - top)
    }

    /**
     * Rotates, scales, and crops [source] so the two eye landmarks land at a canonical position
     * — the standard "face alignment" preprocessing step for embedding models trained the same
     * way (MobileFaceNet/FaceNet/ArcFace all are). This is what actually makes the resulting
     * embeddings comparable across frames: without it, two shots of the same person at slightly
     * different head tilts or distances produce embeddings that drift apart even though it's
     * the same face, which is what was causing the same person to fragment into multiple
     * clusters (or, less often, two different people to falsely merge on a coincidentally
     * similar pose). [leftEye]/[rightEye] must be in [source]'s own pixel coordinate space.
     *
     * Returns a [inputSize] x [inputSize] bitmap ready for [generateEmbedding] — callers
     * shouldn't resize it again.
     */
    fun alignFace(source: Bitmap, leftEye: PointF, rightEye: PointF): Bitmap {
        val dx = (rightEye.x - leftEye.x).toDouble()
        val dy = (rightEye.y - leftEye.y).toDouble()
        val eyeDistance = hypot(dx, dy).toFloat()
        if (eyeDistance < 1f) {
            // Degenerate landmarks (e.g. both eyes reported at ~the same point) — bail out to a
            // plain padded crop rather than dividing by ~0 and producing a garbage transform.
            return Bitmap.createScaledBitmap(source, inputSize, inputSize, true)
        }

        val angleDegrees = Math.toDegrees(atan2(dy, dx)).toFloat()
        // Canonical inter-eye distance as a fraction of the output size. ~36% keeps both eyes,
        // nose, and mouth comfortably inside the frame without cropping the chin — matches the
        // loose convention used by common 112x112 ArcFace/MobileFaceNet-style aligners.
        val targetEyeDistance = inputSize * 0.36f
        val scale = targetEyeDistance / eyeDistance

        val eyeCenterX = (leftEye.x + rightEye.x) / 2f
        val eyeCenterY = (leftEye.y + rightEye.y) / 2f

        val matrix = Matrix().apply {
            postTranslate(-eyeCenterX, -eyeCenterY)
            postRotate(-angleDegrees)
            postScale(scale, scale)
            // Eye line sits ~42% down from the top once aligned, centered horizontally — leaves
            // room for forehead above and chin below within the square output.
            postTranslate(inputSize / 2f, inputSize * 0.42f)
        }

        val aligned = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(aligned)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
        canvas.drawBitmap(source, matrix, paint)
        return aligned
    }

    /**
     * Runs the full pipeline on an already-cropped face bitmap (typically the output of
     * [cropWithPadding]): resize to [inputSize] x [inputSize], normalize to [-1, 1], run
     * inference, and return the raw embedding. Callers who want cosine-similarity comparisons
     * across a gallery of embeddings should L2-normalize the result themselves, or rely on
     * [calculateCosineSimilarity], which is scale-invariant regardless.
     */
    fun generateEmbedding(bitmap: Bitmap): FloatArray {
        val inputBuffer = preprocess(bitmap)
        val outputBuffer = Array(1) { FloatArray(embeddingSize) }

        val start = SystemClock.elapsedRealtime()
        interpreter.run(inputBuffer, outputBuffer)
        val elapsedMs = SystemClock.elapsedRealtime() - start
        // Hook a logger here if you want to track on-device inference latency, e.g.:
        // Log.d("FaceEmbeddingExtractor", "Inference took ${elapsedMs}ms")

        return outputBuffer[0]
    }

    /**
     * Resizes [bitmap] to the model's expected input size and packs it into a normalized
     * float [ByteBuffer] in NHWC order with values scaled from [0, 255] to [-1, 1]:
     * `normalized = (pixel / 127.5) - 1.0`.
     */
    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val resized = if (bitmap.width == inputSize && bitmap.height == inputSize) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        }

        val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF

            byteBuffer.putFloat((r / 127.5f) - 1.0f)
            byteBuffer.putFloat((g / 127.5f) - 1.0f)
            byteBuffer.putFloat((b / 127.5f) - 1.0f)
        }

        byteBuffer.rewind()
        return byteBuffer
    }

    /** Releases the underlying TFLite interpreter. Call when the extractor is no longer needed. */
    fun close() {
        interpreter.close()
    }

    companion object {
        /**
         * Cosine similarity between two embedding vectors, in [-1, 1] (in practice usually
         * [0, 1] for face embeddings from the same model family). Higher = more similar.
         * Returns 0f if either vector has zero magnitude or the vectors differ in length.
         */
        fun calculateCosineSimilarity(vectorA: FloatArray, vectorB: FloatArray): Float {
            if (vectorA.size != vectorB.size || vectorA.isEmpty()) return 0f

            var dotProduct = 0.0
            var normA = 0.0
            var normB = 0.0

            for (i in vectorA.indices) {
                dotProduct += vectorA[i] * vectorB[i]
                normA += vectorA[i] * vectorA[i]
                normB += vectorB[i] * vectorB[i]
            }

            val denominator = sqrt(normA) * sqrt(normB)
            if (denominator == 0.0) return 0f

            return (dotProduct / denominator).toFloat()
        }
    }
}
