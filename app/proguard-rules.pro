# Add project specific ProGuard rules here.
# minifyEnabled is off for debug/release in this build, so these are unused for now,
# but kept in place in case you flip isMinifyEnabled = true later.

# TensorFlow Lite
-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.gpu.** { *; }

# ML Kit
-keep class com.google.mlkit.** { *; }
-keep class com.google.android.gms.internal.mlkit_vision_face.** { *; }
