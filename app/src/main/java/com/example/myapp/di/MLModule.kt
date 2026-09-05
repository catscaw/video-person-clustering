package com.example.myapp.di

import android.content.Context
import com.example.myapp.ml.embedding.FaceEmbeddingExtractor
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Provides ML helper classes as app-scoped singletons so the TFLite interpreter (which is
 * relatively expensive to construct — it maps and validates the whole model file) is loaded
 * once per process, not once per video processed.
 *
 * NOTE: VideoProcessorViewModel currently constructs FaceEmbeddingExtractor directly rather
 * than injecting it — wire this module in if you switch to constructor injection instead.
 */
@Module
@InstallIn(SingletonComponent::class)
object MLModule {

    @Provides
    @Singleton
    fun provideFaceEmbeddingExtractor(@ApplicationContext context: Context): FaceEmbeddingExtractor {
        return FaceEmbeddingExtractor(
            context = context,
            modelFileName = "mobilefacenet.tflite" // must exist under app/src/main/assets/
        )
    }
}
