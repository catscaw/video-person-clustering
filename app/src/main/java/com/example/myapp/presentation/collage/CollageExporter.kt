package com.example.myapp.presentation.collage

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.activity.ComponentActivity
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.platform.ViewCompositionStrategy
import androidx.core.content.FileProvider
import androidx.savedstate.setViewTreeSavedStateRegistryOwner
import androidx.lifecycle.setViewTreeLifecycleOwner
import androidx.lifecycle.setViewTreeViewModelStoreOwner
import kotlinx.coroutines.suspendCancellableCoroutine
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * Renders a Compose UI hierarchy off-screen into a fixed-resolution [Bitmap], and provides
 * utilities to save that bitmap to the device gallery or launch the system share sheet.
 *
 * Export approach: this attaches a real (but invisible, non-interactive) [ComposeView] to the
 * activity's content view so it goes through a normal measure/layout/draw pass at an exact
 * pixel size, then draws that view into a [Canvas] backed by the output [Bitmap], then detaches
 * it. This is the reliable, dependency-free way to rasterize Compose content at a target
 * resolution regardless of the actual device screen size/density. (Newer Compose versions also
 * expose `GraphicsLayer.toImageBitmap()` capture for on-screen composables already being
 * displayed — that's a good fit if you're capturing a view the user is already looking at, but
 * this off-screen approach is more predictable when you need an exact fixed export resolution
 * like 1080x1920 independent of what's currently on screen.)
 */
object CollageExporter {

    /**
     * Composes [content] off-screen at exactly [widthPx] x [heightPx] and returns the
     * rendered result as an ARGB_8888 [Bitmap]. Must be called from the main thread; suspends
     * until the view has been measured, laid out, and drawn at least once.
     */
    suspend fun captureToBitmap(
        activity: ComponentActivity,
        widthPx: Int = 1080,
        heightPx: Int = 1920,
        content: @Composable () -> Unit
    ): Bitmap = suspendCancellableCoroutine { continuation ->
        val rootContainer = activity.findViewById<ViewGroup>(android.R.id.content)

        val composeView = ComposeView(activity).apply {
            setViewCompositionStrategy(ViewCompositionStrategy.DisposeOnDetachedFromWindow)
            // Off-screen, but still part of the window hierarchy so it actually measures/draws.
            alpha = 0f
            isClickable = false
            isFocusable = false
        }

        // Compose needs these owners to compose/recompose; borrow them from the hosting Activity.
        composeView.setViewTreeLifecycleOwner(activity)
        composeView.setViewTreeViewModelStoreOwner(activity)
        composeView.setViewTreeSavedStateRegistryOwner(activity)

        composeView.setContent { content() }

        val layoutParams = FrameLayout.LayoutParams(widthPx, heightPx)
        rootContainer.addView(composeView, layoutParams)

        fun cleanupAndRemove() {
            rootContainer.removeView(composeView)
        }

        // Wait for a real layout pass (composition + measure + layout) before drawing, then
        // detach. Using the view's own layout-change signal is more reliable than a fixed delay.
        composeView.viewTreeObserver.addOnGlobalLayoutListener(object : android.view.ViewTreeObserver.OnGlobalLayoutListener {
            override fun onGlobalLayout() {
                if (composeView.width <= 0 || composeView.height <= 0) return
                composeView.viewTreeObserver.removeOnGlobalLayoutListener(this)

                try {
                    composeView.measure(
                        View.MeasureSpec.makeMeasureSpec(widthPx, View.MeasureSpec.EXACTLY),
                        View.MeasureSpec.makeMeasureSpec(heightPx, View.MeasureSpec.EXACTLY)
                    )
                    composeView.layout(0, 0, widthPx, heightPx)

                    val bitmap = Bitmap.createBitmap(widthPx, heightPx, Bitmap.Config.ARGB_8888)
                    val canvas = Canvas(bitmap)
                    composeView.draw(canvas)

                    cleanupAndRemove()
                    if (continuation.isActive) continuation.resume(bitmap)
                } catch (t: Throwable) {
                    cleanupAndRemove()
                    if (continuation.isActive) continuation.resumeWithException(t)
                }
            }
        })

        continuation.invokeOnCancellation { cleanupAndRemove() }
    }

    /**
     * Saves [bitmap] into the device's Pictures/Collages gallery folder via [MediaStore].
     * Handles both scoped storage (API 29+) and the legacy pre-scoped-storage path (API 26–28,
     * which requires the `WRITE_EXTERNAL_STORAGE` permission declared with
     * `android:maxSdkVersion="28"` as set up earlier). Returns the resulting content [Uri], or
     * null if the write failed.
     */
    fun saveBitmapToGallery(
        context: Context,
        bitmap: Bitmap,
        displayName: String = "collage_${System.currentTimeMillis()}.jpg"
    ): android.net.Uri? {
        val resolver = context.contentResolver

        val contentValues = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/Collages")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
        }

        val collectionUri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        val itemUri = resolver.insert(collectionUri, contentValues) ?: return null

        return try {
            resolver.openOutputStream(itemUri)?.use { outputStream: OutputStream ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 95, outputStream)
            } ?: return null

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                contentValues.clear()
                contentValues.put(MediaStore.Images.Media.IS_PENDING, 0)
                resolver.update(itemUri, contentValues, null, null)
            }
            itemUri
        } catch (t: Throwable) {
            resolver.delete(itemUri, null, null)
            null
        }
    }

    /**
     * Writes [bitmap] to a shareable cache file and launches the standard Android share sheet
     * via [FileProvider] + `ACTION_SEND`. Requires a `<provider>` entry for [authority] in your
     * manifest — see the file_paths.xml / manifest snippet in the kdoc below.
     *
     * ```xml
     * <!-- AndroidManifest.xml, inside <application> -->
     * <provider
     *     android:name="androidx.core.content.FileProvider"
     *     android:authorities="${applicationId}.fileprovider"
     *     android:exported="false"
     *     android:grantUriPermissions="true">
     *     <meta-data
     *         android:name="android.support.FILE_PROVIDER_PATHS"
     *         android:resource="@xml/file_paths" />
     * </provider>
     *
     * <!-- res/xml/file_paths.xml -->
     * <paths>
     *     <cache-path name="shared_images" path="images/" />
     * </paths>
     * ```
     */
    fun shareBitmap(
        context: Context,
        bitmap: Bitmap,
        authority: String = "${context.packageName}.fileprovider",
        chooserTitle: String = "Share collage"
    ) {
        val cacheDir = File(context.cacheDir, "images").apply { mkdirs() }
        val shareFile = File(cacheDir, "shared_collage_${System.currentTimeMillis()}.jpg")

        FileOutputStream(shareFile).use { outputStream ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, outputStream)
        }

        val contentUri = FileProvider.getUriForFile(context, authority, shareFile)

        val shareIntent = Intent(Intent.ACTION_SEND).apply {
            type = "image/jpeg"
            putExtra(Intent.EXTRA_STREAM, contentUri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            if (context !is android.app.Activity) {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
        }

        val chooserIntent = Intent.createChooser(shareIntent, chooserTitle).apply {
            if (context !is android.app.Activity) {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
        }

        context.startActivity(chooserIntent)
    }
}
