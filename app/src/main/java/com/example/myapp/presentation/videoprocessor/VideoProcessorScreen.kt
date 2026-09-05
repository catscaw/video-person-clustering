package com.example.myapp.presentation.videoprocessor

import android.content.Context
import android.content.ContextWrapper
import android.net.Uri
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.myapp.presentation.collage.CollageExporter
import com.example.myapp.presentation.collage.CollageView

/** Shared with [com.example.myapp.presentation.collage.CollageView] so the picker, processing,
 * and results screens all read as one continuous pastel "scrapbook" experience rather than the
 * collage being the only styled piece inside an otherwise default Material screen. */
private val CollagePeach = Color(0xFFF5C4B3)
private val CollagePink = Color(0xFFF4C0D1)
private val CollageLavender = Color(0xFFCECBF6)
private val CollageInk = Color(0xFF4A1B0C)
private val CollageCoral = Color(0xFFD85A30)
private val CollageBackground = Brush.linearGradient(colors = listOf(CollagePeach, CollagePink, CollageLavender))

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun VideoProcessorScreen(
    modifier: Modifier = Modifier,
    viewModel: VideoProcessorViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val activity = remember(context) { context.findComponentActivity() }

    val videoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        // ViewModel.processVideo no longer takes an Activity at all — it only ever touches
        // the injected application Context, so there's nothing here to leak.
        if (uri != null) {
            viewModel.processVideo(uri)
        }
    }

    // The Screen — not the ViewModel — owns the one step that actually needs a live Activity:
    // rendering the off-screen collage. Keyed specifically on the ClusteringCompleted instance
    // (not on `uiState` as a whole) so this effect does NOT restart on every Processing
    // progress tick — only when clustering actually finishes.
    val clusteringCompletedState = uiState as? VideoProcessorUiState.ClusteringCompleted
    LaunchedEffect(clusteringCompletedState) {
        val state = clusteringCompletedState ?: return@LaunchedEffect
        val hostActivity = activity
        if (hostActivity == null) {
            // No Activity available to host the capture (shouldn't normally happen for a
            // full-screen destination) — fail safely back to Idle rather than getting stuck.
            viewModel.onCollageRenderFailed()
            return@LaunchedEffect
        }
        try {
            val collageBitmap = CollageExporter.captureToBitmap(hostActivity) {
                CollageView(people = state.results)
            }
            viewModel.onCollageRendered(collageBitmap)
        } catch (t: Throwable) {
            viewModel.onCollageRenderFailed()
        }
    }

    Scaffold(
        modifier = modifier.fillMaxSize(),
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = "Video Collage",
                        fontStyle = FontStyle.Italic,
                        fontWeight = FontWeight.Medium
                    )
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color.Transparent,
                    titleContentColor = CollageInk
                )
            )
        }
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .padding(paddingValues)
                .fillMaxSize()
        ) {
            when (val state = uiState) {
                is VideoProcessorUiState.Idle -> IdleContent(
                    onPickVideoClick = { videoPickerLauncher.launch("video/*") },
                    modifier = Modifier.fillMaxSize()
                )
                is VideoProcessorUiState.Processing -> ProcessingContent(
                    progress = state.progress,
                    currentStep = state.currentStep,
                    modifier = Modifier.fillMaxSize()
                )
                is VideoProcessorUiState.ClusteringCompleted -> {
                    // Rendered identically to Processing on purpose: from the user's point of
                    // view this is still "processing", just at its final step (building the
                    // collage). This keeps the visible UI flow to exactly three states even
                    // though the state machine has one extra internal handoff state.
                    ProcessingContent(
                        progress = 0.98f,
                        currentStep = "Building your collage…",
                        modifier = Modifier.fillMaxSize()
                    )
                }
                is VideoProcessorUiState.Completed -> CompletedContent(
                    state = state,
                    onSaveToGalleryClick = { viewModel.saveToGallery(context) },
                    onShareClick = { viewModel.share(context) },
                    onStartOverClick = { viewModel.reset() },
                    modifier = Modifier.fillMaxSize()
                )
            }
        }
    }
}

@Composable
private fun IdleContent(onPickVideoClick: () -> Unit, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .background(CollageBackground)
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Box(
            modifier = Modifier
                .size(72.dp)
                .background(Color.White.copy(alpha = 0.55f), CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                Icons.Filled.PlayArrow,
                contentDescription = null,
                tint = CollageInk,
                modifier = Modifier.size(32.dp)
            )
        }
        Spacer(Modifier.height(20.dp))
        Text(
            text = "make the cast",
            color = CollageInk,
            fontSize = 26.sp,
            fontStyle = FontStyle.Italic,
            fontWeight = FontWeight.Medium
        )
        Spacer(Modifier.height(6.dp))
        Text(
            text = "pick a video and we'll find everyone in it",
            color = CollageInk.copy(alpha = 0.7f),
            fontSize = 14.sp,
            fontWeight = FontWeight.Medium
        )
        Spacer(Modifier.height(24.dp))
        Button(
            onClick = onPickVideoClick,
            shape = RoundedCornerShape(percent = 50),
            colors = ButtonDefaults.buttonColors(containerColor = CollageCoral, contentColor = Color.White)
        ) {
            Text("Choose video")
        }
    }
}

@Composable
private fun ProcessingContent(
    progress: Float,
    currentStep: String,
    modifier: Modifier = Modifier
) {
    val animatedProgress by animateFloatAsState(
        targetValue = progress,
        animationSpec = tween(durationMillis = 250),
        label = "collage_processing_progress"
    )

    Column(
        modifier = modifier
            .background(CollageBackground)
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        LinearProgressIndicator(
            progress = { animatedProgress },
            modifier = Modifier
                .fillMaxWidth()
                .height(6.dp)
                .padding(horizontal = 8.dp),
            color = CollageCoral,
            trackColor = Color.White.copy(alpha = 0.5f)
        )
        Spacer(Modifier.height(16.dp))
        Text(text = currentStep, color = CollageInk, fontSize = 15.sp, fontWeight = FontWeight.Medium)
        Spacer(Modifier.height(4.dp))
        Text(
            text = "${(animatedProgress * 100).toInt()}%",
            color = CollageInk.copy(alpha = 0.6f),
            fontSize = 12.sp,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
private fun CompletedContent(
    state: VideoProcessorUiState.Completed,
    onSaveToGalleryClick: () -> Unit,
    onShareClick: () -> Unit,
    onStartOverClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        CollageView(
            people = state.results,
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
        )

        // Share is the one action most people take here, so it's the sole filled/primary
        // button; Save is a secondary outlined action, and Start Over is de-emphasized as
        // plain text so it doesn't compete visually with the two things people actually came
        // here to do.
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color.White)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Button(
                onClick = onShareClick,
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(percent = 50),
                colors = ButtonDefaults.buttonColors(containerColor = CollageCoral, contentColor = Color.White)
            ) {
                Icon(Icons.Filled.Share, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(8.dp))
                Text("Share")
            }
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedButton(
                    onClick = onSaveToGalleryClick,
                    modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(percent = 50),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = CollageInk)
                ) {
                    Text("Save to gallery")
                }
                TextButton(
                    onClick = onStartOverClick,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Start over", color = CollageInk.copy(alpha = 0.6f))
                }
            }
        }
    }
}

/** Walks the Context/ContextWrapper chain to find the hosting ComponentActivity, since
 * LocalContext.current in Compose is sometimes wrapped (e.g. by a Dialog or theme wrapper). */
private fun Context.findComponentActivity(): ComponentActivity? {
    var current = this
    while (current is ContextWrapper) {
        if (current is ComponentActivity) return current
        current = current.baseContext
    }
    return null
}