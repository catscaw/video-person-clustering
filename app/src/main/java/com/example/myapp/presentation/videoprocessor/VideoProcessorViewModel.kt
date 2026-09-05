package com.example.myapp.presentation.videoprocessor

import android.content.Context
import android.util.Log
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.myapp.ml.clustering.FaceObservation
import com.example.myapp.ml.clustering.PersonClusteringEngine
import com.example.myapp.ml.clustering.PersonResult
import com.example.myapp.ml.embedding.FaceEmbeddingExtractor
import com.example.myapp.ml.facedetection.VideoFrameProcessor
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * The pipeline's states. [ClusteringCompleted] is an internal handoff state, not a new visual
 * state — the Screen renders it identically to [Processing] (see `VideoProcessorScreen`), so
 * from the user's perspective the flow is still exactly Idle -> Processing (spinner/progress) ->
 * Completed (collage). [ClusteringCompleted] exists purely so the Screen — which has a live
 * Activity context, unlike this ViewModel — knows it's time to run the off-screen collage
 * capture and hand the resulting bitmap back.
 */
sealed interface VideoProcessorUiState {
    data object Idle : VideoProcessorUiState
    data class Processing(val progress: Float, val currentStep: String) : VideoProcessorUiState
    data class ClusteringCompleted(val results: List<PersonResult>) : VideoProcessorUiState
    data class Completed(val results: List<PersonResult>, val collageBitmap: Bitmap) : VideoProcessorUiState
}

/**
 * Runs frame extraction, face detection, embedding generation, and person clustering —
 * everything that is pure CPU/ML work with no Android View dependency. It deliberately does
 * NOT render or capture the collage itself, and never holds an Activity reference anywhere
 * (field, closure, or otherwise): the only Context this class touches is the injected
 * application Context, which is safe to hold for the ViewModel's entire lifetime.
 *
 * Collage rendering is the Screen's responsibility (see `VideoProcessorScreen`'s
 * `LaunchedEffect` on [VideoProcessorUiState.ClusteringCompleted]), since only the Screen has
 * a live Activity to host the off-screen `ComposeView` that `CollageExporter` needs.
 */
@HiltViewModel
class VideoProcessorViewModel @Inject constructor(
    @ApplicationContext private val appContext: Context
) : ViewModel() {

    private val _uiState = MutableStateFlow<VideoProcessorUiState>(VideoProcessorUiState.Idle)
    val uiState: StateFlow<VideoProcessorUiState> = _uiState.asStateFlow()

    private var processingJob: Job? = null

    /**
     * Runs frame extraction -> face detection -> embedding -> clustering, entirely off the UI
     * thread on [Dispatchers.Default]. Ends in [VideoProcessorUiState.ClusteringCompleted],
     * NOT [VideoProcessorUiState.Completed] — this class has no way to produce a collage
     * bitmap itself, by design.
     */
    fun processVideo(videoUri: Uri) {
        processingJob?.cancel()
        processingJob = viewModelScope.launch(Dispatchers.Default) {
            var embeddingExtractor: FaceEmbeddingExtractor? = null
            try {
                _uiState.value = VideoProcessorUiState.Processing(0f, "Reading video…")

                val frameProcessor = VideoFrameProcessor()
                embeddingExtractor = FaceEmbeddingExtractor(appContext)
                val clusteringEngine = PersonClusteringEngine()

                val frameIntervalMs = 300L
                val estimatedFrameCount = estimateFrameCount(appContext, videoUri, frameIntervalMs)

                val observations = mutableListOf<FaceObservation>()
                var processedFrameCount = 0

                frameProcessor.extractFrames(appContext, videoUri, frameIntervalMs).collect { frame ->
                    frame.faces.forEach { detectedFace ->
                        // Prefer an eye-aligned crop (consistent pose/scale -> much more
                        // comparable embeddings across frames); fall back to a plain padded
                        // crop of the tight ML Kit box on the rare frame where ML Kit couldn't
                        // find both eyes (e.g. a hard profile view).
                        val leftEye = detectedFace.leftEyePosition
                        val rightEye = detectedFace.rightEyePosition
                        val faceCrop = if (
                            leftEye != null && rightEye != null &&
                            kotlin.math.abs(detectedFace.headEulerAngleY) <= 35f
                        ) {
                            embeddingExtractor.alignFace(detectedFace.fullFrameBitmap, leftEye, rightEye)
                        } else {
                            embeddingExtractor.cropWithPadding(
                                detectedFace.fullFrameBitmap,
                                detectedFace.boundingBox,
                                otherFaceBoxesInFrame = detectedFace.otherFaceBoxesInFrame
                            )
                        }
                        val embedding = embeddingExtractor.generateEmbedding(faceCrop)
                        observations += FaceObservation(detectedFace, embedding)
                    }
                    processedFrameCount++
                    val progress = (processedFrameCount.toFloat() / estimatedFrameCount).coerceIn(0f, 0.9f)
                    _uiState.value = VideoProcessorUiState.Processing(progress, "Analyzing faces…")
                }

                _uiState.value = VideoProcessorUiState.Processing(0.95f, "Grouping people…")
                val results = clusteringEngine.cluster(observations)

                _uiState.value = VideoProcessorUiState.ClusteringCompleted(results)
            } catch (cancellation: CancellationException) {
                throw cancellation
            } catch (error: Throwable) {
                Log.e("VideoProcessor", "VIDEO PROCESSING FAILED", error)
                _uiState.value = VideoProcessorUiState.Idle
            } finally {
                embeddingExtractor?.close()
            }
        }
    }

    /**
     * Called by the Screen once it has rendered [VideoProcessorUiState.ClusteringCompleted]
     * into a bitmap via `CollageExporter`. Only accepted while still in that exact state, so a
     * stale/late callback (e.g. from a cancelled or superseded run) can't clobber a newer state.
     */
    fun onCollageRendered(collageBitmap: Bitmap) {
        viewModelScope.launch {
            val current = _uiState.value
            if (current is VideoProcessorUiState.ClusteringCompleted) {
                _uiState.value = VideoProcessorUiState.Completed(current.results, collageBitmap)
            }
        }
    }

    /**
     * Called by the Screen if collage capture itself fails (e.g. the Activity was destroyed
     * mid-capture). Falls back to Idle rather than introducing a dedicated Error state.
     */
    fun onCollageRenderFailed() {
        viewModelScope.launch {
            if (_uiState.value is VideoProcessorUiState.ClusteringCompleted) {
                _uiState.value = VideoProcessorUiState.Idle
            }
        }
    }

    /** Cancels any in-flight run and returns to [VideoProcessorUiState.Idle]. */
    fun reset() {
        processingJob?.cancel()
        _uiState.value = VideoProcessorUiState.Idle
    }

    fun saveToGallery(context: Context) {
        val completed = _uiState.value as? VideoProcessorUiState.Completed ?: return
        viewModelScope.launch(Dispatchers.IO) {
            com.example.myapp.presentation.collage.CollageExporter.saveBitmapToGallery(
                context,
                completed.collageBitmap
            )
        }
    }

    fun share(context: Context) {
        val completed = _uiState.value as? VideoProcessorUiState.Completed ?: return
        viewModelScope.launch(Dispatchers.IO) {
            com.example.myapp.presentation.collage.CollageExporter.shareBitmap(
                context,
                completed.collageBitmap
            )
        }
    }

    private fun estimateFrameCount(context: Context, videoUri: Uri, frameIntervalMs: Long): Long {
        val retriever = MediaMetadataRetriever()
        return try {
            retriever.setDataSource(context, videoUri)
            val durationMs = retriever
                .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: frameIntervalMs
            (durationMs / frameIntervalMs).coerceAtLeast(1L)
        } catch (t: Throwable) {
            1L
        } finally {
            retriever.release()
        }
    }

    override fun onCleared() {
        super.onCleared()
        processingJob?.cancel()
    }
}
