package com.example.myapp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import com.example.myapp.presentation.videoprocessor.VideoProcessorScreen
import dagger.hilt.android.AndroidEntryPoint

/**
 * Single-Activity host. Must be `@AndroidEntryPoint` so `hiltViewModel()` inside
 * [VideoProcessorScreen] can resolve [com.example.myapp.presentation.videoprocessor.VideoProcessorViewModel].
 * This is also the live Activity that [com.example.myapp.presentation.collage.CollageExporter]
 * attaches its off-screen capture view to.
 */
@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    VideoProcessorScreen()
                }
            }
        }
    }
}
