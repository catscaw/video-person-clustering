# Video → Face Collage Pipeline

An on-device Android app that turns a portrait video into an Instagram-Story-style
collage containing one tile per unique person.

The app detects faces throughout the video, generates face embeddings, groups detections
belonging to the same person, counts their separate appearances, selects a representative
frame, and generates a shareable collage — all processed locally on the device.

## Demo

> Add screenshots or a short demo video here.

| Video Input | Generated Collage |
|-------------|-------------------|
| Add screenshot | Add screenshot |

## Features

- 🎥 Select a portrait video from the device
- 👤 Detect multiple faces across video frames
- 🧠 Generate face embeddings using MobileFaceNet
- 🔗 Group detections belonging to the same person
- ⏱️ Count separate appearances of each person
- 🖼️ Select a representative frame for each person
- ✨ Generate an Instagram-Story-style collage
- 📱 Fully on-device processing
- 🔒 No video or face data needs to be uploaded to a server
- 💾 Save the generated collage to the gallery
- 📤 Share the collage using Android's sharing system

## How It Works

```text
                    Portrait Video
                          │
                          ▼
                Frame Extraction
                          │
                          ▼
                  Face Detection
                          │
                          ▼
                Face Alignment
                          │
                          ▼
                 Face Embedding
                          │
                          ▼
                  Person Clustering
                          │
                          ▼
               Appearance Counting
                          │
                          ▼
             Representative Frame
                          │
                          ▼
                Collage Generation
                          │
                    ┌─────┴─────┐
                    ▼           ▼
                  Save        Share
1. Frame Extraction

The video is sampled at regular intervals using MediaMetadataRetriever.
Each sampled frame is passed to the face detection pipeline.

2. Face Detection

Google ML Kit Face Detection identifies faces and provides additional information such as:

Bounding boxes
Eye positions
Eye-open probabilities
Head rotation
Face quality/sharpness information

Landmark detection is enabled so detected faces can be aligned before generating embeddings.

3. Face Alignment

Detected faces are not sent directly to the embedding model.

Eye landmarks are used to rotate and scale the face into a canonical 112 × 112
representation. This reduces embedding variation caused by head tilt, scale, and position.

If both eye landmarks are unavailable, the pipeline falls back to a padded face crop.

4. Face Embeddings

Each aligned face is converted into a numerical embedding using MobileFaceNet.

112 × 112 × 3 image
        │
        ▼
   MobileFaceNet
        │
        ▼
192-dimensional embedding

Embeddings are compared using cosine similarity.

5. Person Clustering

Detections are processed chronologically and assigned to existing person clusters when
their embedding is sufficiently similar.

The clustering stage also contains additional merge passes designed to recover identity
fragments caused by changes in pose, lighting, framing, or camera shots.

Each cluster represents a candidate unique person.

6. Appearance Counting

Once detections have been grouped into people, timestamps are used to determine separate
appearances.

A new appearance is counted when the person has been absent for longer than the configured
appearance gap.

For example:

Person visible ────────────────┐
                               │
                    1 appearance
                               │
Person disappears              │
                               │
              gap > threshold  │
                               ▼
Person visible ────────────────┐
                               │
                    2 appearances

Two people visible during the same continuous segment are counted as one appearance each.

7. Representative Frame Selection

Each person cluster can contain many detections from different frames.

The best representative frame is selected using a combination of:

Face sharpness
Face framing
Eye openness
Head pose

Frames containing another detected face inside the representative crop are rejected so
that a person's tile is less likely to contain a second visible person.

The final image is cropped from the original full-resolution video frame rather than from
the small face bounding box, which helps preserve image quality when the collage is rendered.

8. Collage Generation

The final collage is rendered as an Instagram-Story-style composition with:

Person tiles
Appearance counts
Ranking badges
Rounded cards
Shadows and borders
Background styling
Text overlays

The resulting bitmap can be saved to the device gallery or shared directly.

Architecture
Compose UI
   │
   ▼
VideoProcessorViewModel
   │
   ├── VideoFrameProcessor
   │       └── ML Kit Face Detection
   │
   ├── FaceEmbeddingExtractor
   │       ├── Face Alignment
   │       └── MobileFaceNet
   │
   └── PersonClusteringEngine
           ├── Similarity Matching
           ├── Cluster Merging
           ├── Appearance Counting
           └── Representative Shot Selection
                    │
                    ▼
              CollageExporter
                    │
                    ▼
                CollageView

The expensive processing stages run off the main thread. Only the final UI capture/rendering
step needs to return to the main thread.

Tech Stack
Component	Technology
Language	Kotlin
UI	Jetpack Compose
Face Detection	Google ML Kit
Face Recognition	MobileFaceNet
ML Runtime	TensorFlow Lite
Dependency Injection	Hilt
Video Processing	MediaMetadataRetriever
Image Processing	Android Bitmap APIs
Minimum SDK	API 26+
Target SDK	API 34
Project Structure
app/
├── src/main/
│   ├── assets/
│   │   └── mobilefacenet.tflite
│   │
│   ├── java/com/example/myapp/
│   │   ├── MainActivity.kt
│   │   ├── VideoCollageApplication.kt
│   │   │
│   │   ├── di/
│   │   │   └── MLModule.kt
│   │   │
│   │   ├── ml/
│   │   │   ├── clustering/
│   │   │   │   └── PersonClusteringEngine.kt
│   │   │   ├── embedding/
│   │   │   │   └── FaceEmbeddingExtractor.kt
│   │   │   ├── facedetection/
│   │   │   │   └── VideoFrameProcessor.kt
│   │   │   └── geometry/
│   │   │       └── FaceCropGeometry.kt
│   │   │
│   │   └── presentation/
│   │       ├── collage/
│   │       │   ├── CollageExporter.kt
│   │       │   └── CollageView.kt
│   │       │
│   │       └── videoprocessor/
│   │           ├── VideoProcessorScreen.kt
│   │           └── VideoProcessorViewModel.kt
│   │
│   └── AndroidManifest.xml
│
├── build.gradle.kts
├── settings.gradle.kts
├── gradlew
└── gradlew.bat
Build and Run
Requirements
Android Studio
JDK 17
Android SDK
Android SDK Platform 34
Android Build Tools
Clone
git clone https://github.com/catscaw/video-person-clustering.git
cd video-person-clustering
Build

Linux/macOS:

./gradlew assembleDebug

Windows:

.\gradlew.bat assembleDebug

Or open the project in Android Studio and run the app configuration.

The generated debug APK will be located at:

app/build/outputs/apk/debug/app-debug.apk
Embedding Model

The project uses a MobileFaceNet TFLite model with:

Input:  [1, 112, 112, 3]
Output: [1, 192]

The model is bundled at:

app/src/main/assets/mobilefacenet.tflite

FaceEmbeddingExtractor is responsible for preprocessing the input image and running
TFLite inference.

Performance Considerations

The pipeline performs several computationally expensive operations for every sampled
video frame:

Video decoding
      ↓
Face detection
      ↓
Face alignment
      ↓
TFLite inference
      ↓
Clustering

Processing is performed off the main UI thread to keep the application responsive.

The current implementation samples the video periodically rather than processing every
single frame, trading some temporal precision for substantially lower processing cost.

Current Limitations
Face recognition quality depends on the quality of the source video.
Extremely small, heavily occluded, or profile faces may not be detected reliably.
Face embeddings can still drift under extreme lighting, pose, blur, or occlusion.
Greedy clustering is not guaranteed to find the globally optimal grouping.
Very short appearances may be filtered depending on the configured cluster requirements.
Processing time increases with video duration and the number of detected faces.
Future Improvements

Potential improvements include:

More robust identity clustering
Better handling of extreme pose changes
Automatic selection of optimal frame sampling intervals
GPU/NPU-assisted inference where available
Progress estimation during video processing
Improved collage layouts for larger groups
Better duplicate-identity recovery
More sophisticated representative-frame ranking
Video preview and processing controls
Privacy

Face detection and embedding generation are performed locally on the Android device.

The application does not require a backend service for the core video-processing pipeline.


Author

Dhruvi