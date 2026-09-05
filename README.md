# Video → Face Collage Pipeline

An on-device Android app that processes a portrait video, detects every face, groups
appearances of the same person via face embeddings + clustering, and builds a shareable
Instagram-Story-style collage — one tile per unique person, with an appearance count.

## What changed to make this buildable/runnable

The version this README used to describe was just a set of loose Kotlin source files — no
Gradle project, no manifest, no `MainActivity`, and no embedding model file, so it couldn't
actually be built or run. The following was added/fixed:

1. **Full Gradle project scaffolding** — `settings.gradle.kts`, root/app `build.gradle.kts`,
   `gradlew`/`gradlew.bat` + wrapper jar, `AndroidManifest.xml`, `MainActivity.kt`,
   `VideoCollageApplication.kt` (`@HiltAndroidApp`), `strings.xml`. None of this existed before.
2. **Blurry collage tiles fixed** — `PersonClusteringEngine.selectRepresentativeShot()` used to
   return the *tight* ML Kit face bounding-box crop (often <200px), which `CollageView` then
   stretched to fill a large tile, producing the blurry/blocky tiles you saw. It now crops a
   generous (2.5x) head-and-shoulders region out of the full-resolution source frame instead
   (`PersonClusteringEngine.cropGenerously`).
3. **Embedding model wired up** — `app/src/main/assets/mobilefacenet.tflite` was missing
   entirely (the code referenced it but no file existed, so the app would crash the instant it
   tried to process a video). A real single-face MobileFaceNet TFLite model (float32,
   `[1,112,112,3]` in → `[1,192]` L2-normalized embeddings out — verified locally with a test
   inference before shipping it here) is now bundled. See "Embedding model" below for
   provenance and how to swap it.
4. **Face crops were unaligned before embedding — this was the main accuracy problem.**
   `VideoProcessorViewModel` was feeding the raw, tight ML Kit bounding-box crop straight into
   `generateEmbedding()`. MobileFaceNet-family models are trained on face-aligned crops (eyes
   level, canonical scale/position); an un-aligned crop where head tilt/distance varies frame to
   frame produces embeddings that drift for the *same* person, which is what was causing one
   person to fragment into several clusters (and occasionally two different people to merge on a
   coincidentally similar head pose). Fixed by:
   - Turning on ML Kit's landmark detection (`LANDMARK_MODE_ALL`, it was `NONE`) so eye positions
     are available (`VideoFrameProcessor`, `DetectedFaceInstance.leftEyePosition`/`rightEyePosition`).
   - Adding `FaceEmbeddingExtractor.alignFace()` — rotates/scales/translates the face via eye
     landmarks to a canonical 112×112 pose (standard ArcFace/MobileFaceNet-style alignment)
     before embedding.
   - Wiring this into `VideoProcessorViewModel`, with `cropWithPadding()` (previously dead code —
     defined but never called) as the fallback for the rare frame where ML Kit can't find both
     eyes (hard profile shots).
   - Also set `setMinFaceSize(0.08f)` so tiny/far-away detections (which mostly just added noisy,
     low-resolution junk clusters) get filtered out at detection time.
5. **Collage visuals overhauled** — `CollageView` previously rendered on a flat black
   background with plain rounded-rect tiles and a same-style pill on every tile regardless of
   the photo underneath (so the appearance count could be unreadable on a bright shot). It now
   has a subtle gradient background, a title/count header, per-tile drop shadow + hairline
   border ("card" look), a bottom scrim gradient behind each tile so the appearance pill stays
   legible on any photo, and a small colored rank badge (`#1`, `#2`, …) cycling through an accent
   palette so a multi-person collage reads as a deliberate design rather than a plain grid.

## Build & run

```bash
git clone <this repo>
cd VideoCollagePipeline
./gradlew assembleDebug
```

Or open the project root in Android Studio (Koala/Ladybug or newer), let it sync, and run the
`app` configuration on a device/emulator running API 26+.

The debug APK lands at `app/build/outputs/apk/debug/app-debug.apk`.

**Requirements:** JDK 17, Android SDK with API 34 + build-tools installed (Android Studio
manages this for you; command-line builds need `ANDROID_HOME`/`local.properties` pointing at an
SDK — Android Studio's first sync will create `local.properties` for you automatically).

## Pipeline architecture

```
VideoProcessorScreen (video picker, Compose UI)
        │  Uri
        ▼
VideoProcessorViewModel.processVideo(uri)     [Dispatchers.Default — off the main thread]
        │
        ├─▶ VideoFrameProcessor.extractFrames(...)   → Flow<FrameData>
        │        MediaMetadataRetriever samples a frame every 300ms; ML Kit FaceDetector
        │        (PERFORMANCE_MODE_ACCURATE, CLASSIFICATION_MODE_ALL) runs per frame.
        │
        ├─▶ per DetectedFaceInstance:
        │        FaceEmbeddingExtractor.alignFace(...) → eye-aligned 112x112 crop
        │        (falls back to cropWithPadding(...) if eye landmarks are missing)
        │        FaceEmbeddingExtractor.generateEmbedding(alignedCrop) → 192-dim FloatArray
        │        (TFLite MobileFaceNet, 112x112 input, values normalized to [-1, 1])
        │
        ├─▶ PersonClusteringEngine.cluster(observations) → List<PersonResult>
        │        online greedy cosine-similarity clustering + appearance counting +
        │        representative-shot selection (see below)
        │
        ▼
CollageExporter.captureToBitmap(activity) { CollageView(results) }   [Dispatchers.Main]
        │
        ▼
Save to gallery (MediaStore) / share (FileProvider + ACTION_SEND)
```

All frame decoding, face detection, embedding inference, and clustering run on
`Dispatchers.Default`/`Dispatchers.IO` — never the main thread. Only the final Compose capture
step touches the main thread, since rendering has to.

## Embedding model

**Model:** MobileFaceNet, TFLite float32, 112×112×3 input → 192-dim L2-normalized embedding
output, batch size 1 (`app/src/main/assets/mobilefacenet.tflite`, ~5MB).

Sourced from the `hugocornellier/face_detection_tflite` project's bundled MobileFaceNet
conversion (Apache-2.0), which itself derives from the original
[sirius-ai/MobileFaceNet_TF](https://github.com/sirius-ai/MobileFaceNet_TF). Verified locally
before bundling that it loads with input shape `[1,112,112,3]` and produces a unit-norm
`[1,192]` output — this matters because some other public "mobilefacenet.tflite" files floating
around GitHub are actually exported for **pairwise** batch-of-2 comparison
(`[2,112,112,3]` → `[2,192]`) and will crash a batch-1 embedding pipeline like this one.

If you swap in a different model, update `FaceEmbeddingExtractor`'s `inputSize`/`embeddingSize`
constructor params (and `modelFileName` in `di/MLModule.kt` if you rename the asset) to match —
a shape mismatch fails at inference time, not compile time.

## Clustering & similarity threshold

`PersonClusteringEngine` uses single-pass, greedy nearest-centroid cosine-similarity clustering
(assign each face, in timestamp order, to its most similar existing cluster if similarity ≥
threshold, else start a new cluster; centroid = running mean).

- **`similarityThreshold = 0.65`** — chosen empirically as a reasonable middle ground for
  MobileFaceNet-family embeddings on portrait video (faces at varying angles/expressions
  within one clip). If you find two different people getting merged into one cluster, raise it
  (e.g. 0.72–0.78); if the same person keeps splitting into multiple clusters across a video
  (pose/lighting drift), lower it slightly (e.g. 0.55–0.6) instead.
- **`minClusterDetections = 2`** (was `3`) — clusters with fewer than 2 detections are discarded
  as noise (e.g. a single blurry whip-pan frame that briefly resembled a face). Lowered from 3
  because at the default 300ms sample interval, `3` required someone to be on screen for ~900ms
  continuously just to register — long enough to silently drop people who only appear briefly.
  If you start seeing false-positive "phantom" tiles from motion blur, raise it back.
- **`appearanceGapMs = 800`** — a new "appearance" starts once a person hasn't been detected for
  more than 800ms, matching the brief's "continuous visible segment" definition.

### If people are still missing from the collage

Two log tags will tell you exactly where in the pipeline they're being lost — filter Logcat for
either while processing a video:

- **`VideoFrameProcessor`** logs whenever a frame has more than one face. If someone visible in
  the video never shows up in these logs, ML Kit itself isn't detecting their face in any
  frame — usually because they're small/distant, side-on, or partly occluded. Try lowering
  `setMinFaceSize` in `VideoFrameProcessor` (currently `0.08f`) further.
- **`PersonClusteringEngine`** logs one summary line per `cluster()` call: how many raw
  clusters came out of similarity matching, their sizes, and how many survived
  `minClusterDetections`. Many small raw clusters that got filtered out means people *are*
  being detected but only briefly (lower `minClusterDetections` further). Very few raw clusters
  relative to how many people are in the video means embeddings are being merged together
  (raise `similarityThreshold`).

## Representative-shot selection

Every candidate detection in a cluster is scored:

```
score = frontality * 0.3 + eyesOpen * 0.3 + smileProb * 0.2 + sharpness * 0.2
```

- **frontality**: `1 - |headEulerAngleY| / 45`, clamped to [0,1] — how front-facing the head is.
- **eyesOpen**: average of ML Kit's left/right eye-open probabilities.
- **sharpness**: variance-of-Laplacian blur estimate (no OpenCV dependency), normalized against
  the sharpest candidate in that cluster.
- Any candidate where either eye-open probability is below 0.35 is heavily penalized (×0.1) so
  a sharp-but-blinking frame loses to a slightly softer eyes-open one unless every candidate in
  the cluster has closed eyes.

The winning detection is then cropped generously (2.5x the face bounding box, centered on it,
shifted back in-bounds rather than shrunk if it would run off a frame edge) out of the
**full-resolution source frame** — not the tight face box — so collage tiles hold up at the
larger sizes `CollageView` renders them at.

## Known trade-offs / things to be aware of

- Clustering is O(n·k) greedy, not full agglomerative — order-dependent, can under-merge an
  identity whose embedding drifts a lot over a video. A second merge pass over final cluster
  centroids would improve this further if needed.
- `VideoProcessorViewModel` constructs its own `FaceEmbeddingExtractor` per video rather than
  injecting the Hilt-provided singleton from `di/MLModule.kt` — functionally fine, just means
  the TFLite interpreter is reloaded each run rather than reused across the app session.
- No runtime permission request is implemented for the video picker (the `ActivityResultContracts.GetContent()`
  system picker doesn't require one on any API level this targets, so it works as-is, but a
  production app targeting other pickers should still request `READ_MEDIA_VIDEO`/
  `READ_EXTERNAL_STORAGE` explicitly per `Build.VERSION.SDK_INT`).
