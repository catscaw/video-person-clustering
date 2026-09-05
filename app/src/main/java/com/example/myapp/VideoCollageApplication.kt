package com.example.myapp

import android.app.Application
import dagger.hilt.android.HiltAndroidApp

/**
 * Required so Hilt can generate the app-level component that [VideoProcessorViewModel] (via
 * `@HiltViewModel`) and [com.example.myapp.di.MLModule] hang off of. Registered in
 * AndroidManifest.xml as `android:name=".VideoCollageApplication"`.
 */
@HiltAndroidApp
class VideoCollageApplication : Application()
