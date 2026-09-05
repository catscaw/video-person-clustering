package com.example.myapp.ml.clustering

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.example.myapp.ml.embedding.FaceEmbeddingExtractor.Companion.calculateCosineSimilarity
import com.example.myapp.ml.facedetection.DetectedFaceInstance
import java.util.UUID
import kotlin.math.abs

data class FaceObservation(
    val detectedFace: DetectedFaceInstance,
    val embedding: FloatArray
)

data class PersonResult(
    val personId: String,
    val totalAppearances: Int,
    val bestFrameBitmap: Bitmap
)

class PersonClusteringEngine(
    // Higher = harder for two different people to get merged.
    private val similarityThreshold: Float = 0.62f,

    // Keep even people that are only detected once.
    private val minClusterDetections: Int = 3,

    private val appearanceGapMs: Long = 2000L,

    private val minEyeOpenProbability: Float = 0.35f,

    // Require the best match to be clearly better than the second-best match.
    private val similarityMargin: Float = 0.015f,

    // Don't let a cluster contain an unlimited number of old embeddings.
    private val maxReferenceEmbeddings: Int = 10
)



{

    private data class Cluster(
        val id: String,
        val observations: MutableList<FaceObservation>,
        val referenceEmbeddings: MutableList<Pair<FloatArray, Double>>
    )

    fun cluster(observations: List<FaceObservation>): List<PersonResult> {
        if (observations.isEmpty()) return emptyList()

        val sortedByTime = observations.sortedBy {
            it.detectedFace.timestampMs
        }

        val clusters = mutableListOf<Cluster>()

        for (observation in sortedByTime) {

            if (clusters.isEmpty()) {
                clusters.add(
                    newCluster(observation)
                )
                continue
            }


            // Calculate similarity against every reference embedding,
            // not only against a drifting centroid.
            val clusterScores = clusters.map { cluster ->

                val similarities = cluster.referenceEmbeddings
                    .map { (embedding, _) -> calculateCosineSimilarity(embedding, observation.embedding) }
                    .sortedDescending()

                val score = similarities.firstOrNull() ?: 0f

                cluster to score

            }.sortedByDescending { it.second }

            val bestCluster = clusterScores[0].first
            val bestSimilarity = clusterScores[0].second


            val shouldMerge = if (observation.detectedFace.sharpnessScore >= 0.35) {
                bestSimilarity >= similarityThreshold
            } else {
                bestSimilarity >= similarityThreshold + 0.04f
            }

            if (shouldMerge) {
                bestCluster.observations.add(observation)

                // Store this embedding as another reference point.
                // This handles pose/light changes much better than
                // continuously moving a centroid.
                if (observation.detectedFace.sharpnessScore >= 0.55) {
                    bestCluster.referenceEmbeddings.add(
                        observation.embedding.copyOf() to observation.detectedFace.sharpnessScore
                    )

                    if (bestCluster.referenceEmbeddings.size > maxReferenceEmbeddings) {
                        val mostRedundantIndex = bestCluster.referenceEmbeddings.indices.maxByOrNull { i ->
                            val (embI, _) = bestCluster.referenceEmbeddings[i]
                            bestCluster.referenceEmbeddings.indices.filter { it != i }
                                .maxOf { j -> calculateCosineSimilarity(embI, bestCluster.referenceEmbeddings[j].first) }
                        }!!
                        bestCluster.referenceEmbeddings.removeAt(mostRedundantIndex)
                    }
                }   // <-- ADD THIS

                Log.d(
                    TAG,
                    "MERGE person=${bestCluster.id.take(8)} " +
                            "score=${"%.3f".format(bestSimilarity)}"
                )
            } else {
                clusters.add(newCluster(observation))
            }
        }

        mergeSimilarClusters(clusters)
        mergeWithRelaxedThreshold(clusters, 0.56f)
        mergeSmallSimilarClusters(clusters, 0.56f)
        rescueByProximity(clusters)
        rescueByTypicalPosition(clusters)

        // Final pass for identity fragments that were created in separate parts of the video.
        // Only merge clusters that do not appear at the same time and have a reasonably strong
        // face-embedding match.
        mergeTemporallySeparatedDuplicates(clusters)

        val trustedClusters = clusters.filter { it.observations.size >= minClusterDetections }



        Log.d(
            TAG,
            "FINAL: observations=${observations.size}, " +
                    "rawClusters=${clusters.size}, " +
                    "kept=${trustedClusters.size}, " +
                    "sizes=${clusters.map { it.observations.size }.sortedDescending()}"
        )

        return trustedClusters.map { cluster ->
            PersonResult(
                personId = cluster.id,
                totalAppearances = countAppearances(cluster.observations),
                bestFrameBitmap = selectRepresentativeShot(
                    cluster.observations
                )
            )
        }
    }

    private fun newCluster(
        observation: FaceObservation
    ): Cluster {
        return Cluster(
            id = UUID.randomUUID().toString(),
            observations = mutableListOf(observation),
            referenceEmbeddings = mutableListOf(
                observation.embedding.copyOf() to observation.detectedFace.sharpnessScore
            )
        )
    }

    private fun mergeSmallSimilarClusters(
        clusters: MutableList<Cluster>,
        threshold: Float = 0.55f
    ) {
        var changed = true

        while (changed) {
            changed = false

            outer@ for (i in 0 until clusters.size) {
                for (j in i + 1 until clusters.size) {

                    // Only compare small clusters
                    if (clusters[i].observations.size > 6 ||
                        clusters[j].observations.size > 6
                    ) {
                        continue
                    }

                    val similarity = clusters[i].referenceEmbeddings
                        .flatMap { (a, _) ->
                            clusters[j].referenceEmbeddings.map { (b, _) ->
                                calculateCosineSimilarity(a, b)
                            }
                        }
                        .maxOrNull() ?: 0f

                    Log.d(
                        TAG,
                        "SMALL_CLUSTER_COMPARE " +
                                "${clusters[i].observations.size} vs " +
                                "${clusters[j].observations.size} " +
                                "similarity=${"%.3f".format(similarity)}"
                    )

                    if (similarity >= threshold) {

                        Log.d(
                            TAG,
                            "SMALL_CLUSTER_MERGE " +
                                    "similarity=${"%.3f".format(similarity)}"
                        )

                        clusters[i].observations.addAll(
                            clusters[j].observations
                        )

                        clusters[i].referenceEmbeddings.addAll(
                            clusters[j].referenceEmbeddings
                        )

                        while (
                            clusters[i].referenceEmbeddings.size >
                            maxReferenceEmbeddings
                        ) {
                            val worstIndex =
                                clusters[i].referenceEmbeddings.indices
                                    .minBy {
                                        clusters[i].referenceEmbeddings[it].second
                                    }

                            clusters[i].referenceEmbeddings.removeAt(worstIndex)
                        }

                        clusters.removeAt(j)

                        changed = true
                        break@outer
                    }
                }
            }
        }
    }

    private fun mergeSimilarClusters(
        clusters: MutableList<Cluster>
    ) {
        var changed = true

        while (changed) {
            changed = false

            outer@ for (i in 0 until clusters.size) {
                for (j in i + 1 until clusters.size) {

                    val similarities = clusters[i].referenceEmbeddings
                        .flatMap { (a, _) ->
                            clusters[j].referenceEmbeddings.map { (b, _) ->
                                calculateCosineSimilarity(a, b)
                            }
                        }
                        .sortedDescending()

                    if (similarities.isEmpty()) continue

                    val best = similarities.first()
                    val secondBest = similarities.getOrNull(1) ?: best

                    // For already-established clusters, require a strong match.
                    // For smaller clusters, allow a slightly more forgiving match.
                    val smallerSize = minOf(
                        clusters[i].observations.size,
                        clusters[j].observations.size
                    )

                    val threshold = when {
                        smallerSize <= 5 -> 0.58f
                        smallerSize <= 10 -> 0.60f
                        else -> similarityThreshold
                    }

                    Log.d(
                        TAG,
                        "MERGE_CHECK best=${"%.3f".format(best)} " +
                                "second=${"%.3f".format(secondBest)} " +
                                "threshold=${"%.3f".format(threshold)} " +
                                "sizeA=${clusters[i].observations.size} " +
                                "sizeB=${clusters[j].observations.size}"
                    )

                    if (best >= threshold) {

                        clusters[i].observations.addAll(
                            clusters[j].observations
                        )

                        clusters[i].referenceEmbeddings.addAll(
                            clusters[j].referenceEmbeddings
                        )

                        while (
                            clusters[i].referenceEmbeddings.size >
                            maxReferenceEmbeddings
                        ) {
                            val worstIndex =
                                clusters[i].referenceEmbeddings.indices
                                    .minByOrNull {
                                        clusters[i].referenceEmbeddings[it].second
                                    } ?: break

                            clusters[i].referenceEmbeddings.removeAt(worstIndex)
                        }

                        Log.d(
                            TAG,
                            "MERGE_CLUSTER " +
                                    "best=${"%.3f".format(best)} " +
                                    "newSize=${clusters[i].observations.size}"
                        )

                        clusters.removeAt(j)

                        changed = true
                        break@outer
                    }
                }
            }
        }
    }

    /** Shrinks [rect] (a proposed crop) just enough that it no longer overlaps any box in
     *  [neighbors], leaving a small [minGapPx] buffer. Shrinks from whichever edge is closest
     *  to each neighbor, so it only gives up as much space as it actually needs to. */


    private fun rescueByProximity(clusters: MutableList<Cluster>) {
        // A genuine fragment (a pose/expression that just missed the embedding threshold) is
        // almost always small — a handful of observations at most. A cluster with a
        // meaningful number of its own observations (10+) is very likely a real second person
        // who happened to stand near someone else, not a fragment — rescuing those loses a
        // real person from the output. Cap candidates at a small absolute size instead of
        // "smaller half of all clusters", which let exactly that happen.
        val maxCandidateSize = 6
        val established = clusters.filter { it.observations.size > maxCandidateSize }
        val candidates = clusters.filter { it.observations.size <= maxCandidateSize }

        for (candidate in candidates) {
            val candidateFirst = candidate.observations.minBy { it.detectedFace.timestampMs }
            val candidateLast = candidate.observations.maxBy { it.detectedFace.timestampMs }

            val match = established.firstOrNull { target ->
                val closeInSpaceAndTime = target.observations.any { obs ->
                    val timeDelta = kotlin.math.min(
                        kotlin.math.abs(obs.detectedFace.timestampMs - candidateFirst.detectedFace.timestampMs),
                        kotlin.math.abs(obs.detectedFace.timestampMs - candidateLast.detectedFace.timestampMs)
                    )
                    if (timeDelta > 1000L) return@any false

                    val dx = obs.detectedFace.boundingBox.centerX() - candidateFirst.detectedFace.boundingBox.centerX()
                    val dy = obs.detectedFace.boundingBox.centerY() - candidateFirst.detectedFace.boundingBox.centerY()
                    val distance = kotlin.math.hypot(dx.toDouble(), dy.toDouble())
                    distance < obs.detectedFace.boundingBox.width() * 1.5
                }
                if (!closeInSpaceAndTime) return@firstOrNull false

                val similarities = target.referenceEmbeddings
                    .flatMap { (a, _) -> candidate.referenceEmbeddings.map { (b, _) -> calculateCosineSimilarity(a, b) } }
                    .sortedDescending()
                val score = similarities.firstOrNull() ?: 0f

                score >= 0.40f   // was 0.15f — a bit more skeptical, now that size alone can't save us
            }

            if (match != null) {
                Log.d(TAG, "PROXIMITY_MERGE candidateSize=${candidate.observations.size} into targetSize=${match.observations.size}")
                match.observations.addAll(candidate.observations)
                match.referenceEmbeddings.addAll(candidate.referenceEmbeddings)
                clusters.remove(candidate)
            }
        }
    }

    // A cluster with very few observations and only one true "appearance" is far more
// likely to be a fragment of an already-established person (an extreme angle,
// occlusion, or expression that narrowly missed the main threshold) than a brand
// new person who genuinely only appeared once. Give these a second, more lenient
// chance to merge into an established cluster before they reach final output.
    private fun mergeWithRelaxedThreshold(
        clusters: MutableList<Cluster>,
        relaxedThreshold: Float
    ) {
        var changed = true

        while (changed) {
            changed = false

            outer@ for (i in 0 until clusters.size) {
                for (j in i + 1 until clusters.size) {

                    val similarities = clusters[i].referenceEmbeddings
                        .flatMap { (a, _) -> clusters[j].referenceEmbeddings.map { (b, _) -> calculateCosineSimilarity(a, b) } }
                        .sortedDescending()

                    if (similarities.isEmpty()) continue

                    // Same reasoning as mergeSimilarClusters: take the best pair, not an average.
                    val similarity = similarities.first()

                    if (similarity >= relaxedThreshold) {
                        clusters[i].observations.addAll(clusters[j].observations)
                        clusters[i].referenceEmbeddings.addAll(clusters[j].referenceEmbeddings)

                        while (clusters[i].referenceEmbeddings.size > maxReferenceEmbeddings) {
                            val mostRedundantIndex = clusters[i].referenceEmbeddings.indices.maxByOrNull { idx ->
                                val (embI, _) = clusters[i].referenceEmbeddings[idx]
                                clusters[i].referenceEmbeddings.indices.filter { it != idx }
                                    .maxOf { k -> calculateCosineSimilarity(embI, clusters[i].referenceEmbeddings[k].first) }
                            }!!
                            clusters[i].referenceEmbeddings.removeAt(mostRedundantIndex)
                        }

                        clusters.removeAt(j)
                        changed = true
                        break@outer
                    }
                }
            }
        }
    }

    private fun mergeTemporallySeparatedDuplicates(
        clusters: MutableList<Cluster>
    ) {
        /*
         * Final identity-repair pass.
         *
         * The important distinction is:
         *
         * 1. Two clusters visible at DIFFERENT times:
         *    allow a lower embedding threshold because they cannot
         *    be two different people standing beside each other.
         *
         * 2. Two clusters visible at the SAME time:
         *    require a much stronger match AND similar location.
         */

        val temporallySeparatedThreshold = 0.34f
        val simultaneousThreshold = 0.55f

        val simultaneousWindowMs = 800L
        val sameLocationFactor = 1.20

        var changed = true

        while (changed) {
            changed = false

            outer@ for (i in 0 until clusters.size) {
                for (j in i + 1 until clusters.size) {

                    val clusterA = clusters[i]
                    val clusterB = clusters[j]

                    /*
                     * Find the strongest identity match between the two
                     * clusters using ALL retained reference embeddings.
                     */
                    val bestSimilarity =
                        clusterA.referenceEmbeddings
                            .flatMap { (embeddingA, _) ->
                                clusterB.referenceEmbeddings.map { (embeddingB, _) ->
                                    calculateCosineSimilarity(
                                        embeddingA,
                                        embeddingB
                                    )
                                }
                            }
                            .maxOrNull() ?: 0f

                    var temporalOverlap = false
                    var sameLocationDuringOverlap = false

                    /*
                     * Check whether the two clusters are present at
                     * approximately the same moment.
                     */
                    for (a in clusterA.observations) {
                        for (b in clusterB.observations) {

                            val timeDifference =
                                abs(
                                    a.detectedFace.timestampMs -
                                            b.detectedFace.timestampMs
                                )

                            if (timeDifference > simultaneousWindowMs) {
                                continue
                            }

                            temporalOverlap = true

                            val dx =
                                a.detectedFace.boundingBox.centerX() -
                                        b.detectedFace.boundingBox.centerX()

                            val dy =
                                a.detectedFace.boundingBox.centerY() -
                                        b.detectedFace.boundingBox.centerY()

                            val distance =
                                kotlin.math.hypot(
                                    dx.toDouble(),
                                    dy.toDouble()
                                )

                            val faceWidth =
                                minOf(
                                    a.detectedFace.boundingBox.width(),
                                    b.detectedFace.boundingBox.width()
                                ).coerceAtLeast(1)

                            if (distance <= faceWidth * sameLocationFactor) {
                                sameLocationDuringOverlap = true
                                break
                            }
                        }

                        if (sameLocationDuringOverlap) {
                            break
                        }
                    }

                    /*
                     * CASE 1:
                     * No temporal overlap.
                     *
                     * These are separate shots/segments. Allow a much
                     * lower similarity because this is exactly where
                     * the same person tends to get fragmented.
                     */
                    if (!temporalOverlap) {

                        if (bestSimilarity < temporallySeparatedThreshold) {
                            continue
                        }

                        Log.d(
                            TAG,
                            "TEMPORAL_DUPLICATE_MERGE " +
                                    "similarity=${"%.3f".format(bestSimilarity)} " +
                                    "sizeA=${clusterA.observations.size} " +
                                    "sizeB=${clusterB.observations.size}"
                        )

                        clusterA.observations.addAll(
                            clusterB.observations
                        )

                        clusterA.referenceEmbeddings.addAll(
                            clusterB.referenceEmbeddings
                        )

                        while (
                            clusterA.referenceEmbeddings.size >
                            maxReferenceEmbeddings
                        ) {
                            val worstIndex =
                                clusterA.referenceEmbeddings.indices
                                    .minByOrNull {
                                        clusterA.referenceEmbeddings[it].second
                                    } ?: break

                            clusterA.referenceEmbeddings.removeAt(worstIndex)
                        }

                        clusters.removeAt(j)

                        changed = true
                        break@outer
                    }

                    /*
                     * CASE 2:
                     * Both clusters exist at the same time.
                     *
                     * Never use the relaxed threshold here.
                     *
                     * Two different people can have reasonably similar
                     * embeddings, so require a stronger match and also
                     * require their detections to occupy approximately
                     * the same position.
                     */
                    if (
                        bestSimilarity >= simultaneousThreshold &&
                        sameLocationDuringOverlap
                    ) {
                        Log.d(
                            TAG,
                            "SIMULTANEOUS_DUPLICATE_MERGE " +
                                    "similarity=${"%.3f".format(bestSimilarity)} " +
                                    "sizeA=${clusterA.observations.size} " +
                                    "sizeB=${clusterB.observations.size}"
                        )

                        clusterA.observations.addAll(
                            clusterB.observations
                        )

                        clusterA.referenceEmbeddings.addAll(
                            clusterB.referenceEmbeddings
                        )

                        while (
                            clusterA.referenceEmbeddings.size >
                            maxReferenceEmbeddings
                        ) {
                            val worstIndex =
                                clusterA.referenceEmbeddings.indices
                                    .minByOrNull {
                                        clusterA.referenceEmbeddings[it].second
                                    } ?: break

                            clusterA.referenceEmbeddings.removeAt(worstIndex)
                        }

                        clusters.removeAt(j)

                        changed = true
                        break@outer
                    }

                    /*
                     * Otherwise keep the identities separate.
                     */
                }
            }
        }
    }

    private fun countAppearances(
        observations: List<FaceObservation>
    ): Int {

        if (observations.isEmpty()) return 0

        val sortedByTime = observations.sortedBy {
            it.detectedFace.timestampMs
        }

        var appearanceCount = 1

        for (i in 1 until sortedByTime.size) {
            val gap =
                sortedByTime[i].detectedFace.timestampMs -
                        sortedByTime[i - 1].detectedFace.timestampMs

            if (gap > appearanceGapMs) {
                appearanceCount++
            }
        }

        return appearanceCount
    }


    private fun selectRepresentativeShot(
        observations: List<FaceObservation>
    ): Bitmap {

        if (observations.isEmpty()) {
            throw IllegalArgumentException("No observations")
        }

        var bestObservation: FaceObservation? = null
        var bestScore = Double.NEGATIVE_INFINITY

        for (observation in observations) {
            val face = observation.detectedFace
            val frame = face.fullFrameBitmap
            val box = face.boundingBox

            if (box.width() <= 0 || box.height() <= 0) continue
            if (face.sharpnessScore < 0.20) continue

            // EXACT same crop geometry as cropPerson().
            val cropWidth = (box.width() * 1.95f).toInt()
            val cropHeight = (box.height() * 2.70f).toInt()
            val centerX = box.centerX()
            val centerY = box.centerY() + (box.height() * 0.28f)

            val candidateRect = Rect(
                (centerX - cropWidth / 2).coerceIn(0, frame.width),
                (centerY - cropHeight / 2f)
                    .toInt()
                    .coerceIn(0, frame.height),
                (centerX + cropWidth / 2)
                    .coerceIn(0, frame.width),
                (centerY + cropHeight / 2f)
                    .toInt()
                    .coerceIn(0, frame.height)
            )

            // Slightly expand the checking area so we also reject frames
            // where another person's face is just entering the edge.
            val detectionRect = Rect(candidateRect)

            val extraX = (candidateRect.width() * 0.15f).toInt()
            val extraY = (candidateRect.height() * 0.10f).toInt()

            detectionRect.left =
                (detectionRect.left - extraX).coerceAtLeast(0)

            detectionRect.top =
                (detectionRect.top - extraY).coerceAtLeast(0)

            detectionRect.right =
                (detectionRect.right + extraX)
                    .coerceAtMost(frame.width)

            detectionRect.bottom =
                (detectionRect.bottom + extraY)
                    .coerceAtMost(frame.height)

            val containsAnotherFace =
                face.otherFaceBoxesInFrame.any { otherBox ->

                    if (otherBox.width() <= 0 || otherBox.height() <= 0) {
                        return@any false
                    }

                    val intersection = Rect()

                    if (!intersection.setIntersect(
                            detectionRect,
                            otherBox
                        )
                    ) {
                        return@any false
                    }

                    val otherFaceArea =
                        otherBox.width().toFloat() *
                                otherBox.height().toFloat()

                    val intersectionArea =
                        intersection.width().toFloat() *
                                intersection.height().toFloat()

                    val overlapRatio =
                        if (otherFaceArea > 0f) {
                            intersectionArea / otherFaceArea
                        } else {
                            0f
                        }

                    // Reject if a meaningful part of another face is
                    // inside the checking area, or its center is inside it.
                    overlapRatio >= 0.10f ||
                            detectionRect.contains(
                                otherBox.centerX(),
                                otherBox.centerY()
                            )
                }

            if (containsAnotherFace) {
                Log.d(
                    TAG,
                    "REP_REJECT_SECOND_FACE"
                )
                continue
            }

            val leftEye =
                (face.leftEyeOpenProbability ?: 0f).toDouble()

            val rightEye =
                (face.rightEyeOpenProbability ?: 0f).toDouble()

            val eyeScore =
                ((leftEye + rightEye) / 2.0)
                    .coerceIn(0.0, 1.0)

            val poseScore =
                (1.0 - abs(face.headEulerAngleY) / 45.0)
                    .coerceIn(0.0, 1.0)

            val faceAreaRatio =
                (box.width().toDouble() *
                        box.height().toDouble()) /
                        (frame.width.toDouble() *
                                frame.height.toDouble())

            val sharpnessScore =
                face.sharpnessScore
                    .coerceIn(0.0, 1.0)

            // Prefer a well-framed face without heavily penalizing
            // legitimate close-ups.
            val framingScore =
                (1.0 - (faceAreaRatio / 0.45))
                    .coerceIn(0.0, 1.0)

            var score =
            sharpnessScore * 0.65 +
                    framingScore * 0.20 +
                    eyeScore * 0.10 +
                    poseScore * 0.05

            if (
                leftEye < minEyeOpenProbability ||
                rightEye < minEyeOpenProbability
            ) {
                score *= 0.15
            }

            if (score > bestScore) {
                bestScore = score
                bestObservation = observation
            }
        }

        /*
         * Only use a frame that passed the second-face check.
         *
         * If every frame was rejected, choose the frame with the
         * smallest amount of other-face overlap rather than blindly
         * selecting the sharpest frame.
         */
        val selectedObservation =
            bestObservation ?: observations.minByOrNull { observation ->

                val face = observation.detectedFace
                val frame = face.fullFrameBitmap
                val box = face.boundingBox

                if (box.width() <= 0 || box.height() <= 0) {
                    return@minByOrNull Double.POSITIVE_INFINITY
                }

                val cropWidth =
                    (box.width() * 1.95f).toInt()

                val cropHeight =
                    (box.height() * 2.70f).toInt()

                val centerX = box.centerX()
                val centerY =
                    box.centerY() +
                            (box.height() * 0.28f)

                val candidateRect = Rect(
                    (centerX - cropWidth / 2)
                        .coerceIn(0, frame.width),

                    (centerY - cropHeight / 2f)
                        .toInt()
                        .coerceIn(0, frame.height),

                    (centerX + cropWidth / 2)
                        .coerceIn(0, frame.width),

                    (centerY + cropHeight / 2f)
                        .toInt()
                        .coerceIn(0, frame.height)
                )

                face.otherFaceBoxesInFrame.sumOf { otherBox ->

                    val intersection = Rect()

                    if (!intersection.setIntersect(
                            candidateRect,
                            otherBox
                        )
                    ) {
                        return@sumOf 0.0
                    }

                    val area =
                        otherBox.width().toDouble() *
                                otherBox.height().toDouble()

                    if (area <= 0.0) {
                        0.0
                    } else {
                        (
                                intersection.width().toDouble() *
                                        intersection.height().toDouble()
                                ) / area
                    }
                }

            } ?: observations.maxByOrNull {
                it.detectedFace.sharpnessScore
            } ?: observations.first()

        // KEEP YOUR ORIGINAL FRAMING.
        return cropPerson(
            selectedObservation.detectedFace,
            margin = 1.8f
        )
    }


    private fun rescueByTypicalPosition(clusters: MutableList<Cluster>) {
        val maxCandidateSize = 6
        val established = clusters.filter { it.observations.size > maxCandidateSize }
        val candidates = clusters.filter { it.observations.size <= maxCandidateSize }

        for (candidate in candidates) {
            val candidateCenterX = candidate.observations.map { it.detectedFace.boundingBox.centerX() }.average()
            val candidateCenterY = candidate.observations.map { it.detectedFace.boundingBox.centerY() }.average()
            val candidateFaceWidth = candidate.observations.map { it.detectedFace.boundingBox.width() }.average()

            val match = established.firstOrNull { target ->
                val similarities = target.referenceEmbeddings
                    .flatMap { (a, _) -> candidate.referenceEmbeddings.map { (b, _) -> calculateCosineSimilarity(a, b) } }
                    .sortedDescending()
                val score = similarities.firstOrNull() ?: 0f

                // A confident embedding match is trustworthy on its own. A cutaway/insert
                // close-up of the same person (leaning into a mic, a reaction shot, etc.)
                // deliberately breaks position and framing — that's the whole point of the
                // shot — so position/size shouldn't be a hard requirement when the face itself
                // is already a strong match. Only fall back to requiring position/size
                // agreement when the embedding evidence is weaker.
                if (score >= similarityThreshold - 0.08f) return@firstOrNull true

                val targetCenterX = target.observations.map { it.detectedFace.boundingBox.centerX() }.average()
                val targetCenterY = target.observations.map { it.detectedFace.boundingBox.centerY() }.average()
                val targetFaceWidth = target.observations.map { it.detectedFace.boundingBox.width() }.average()

                val distance = kotlin.math.hypot(candidateCenterX - targetCenterX, candidateCenterY - targetCenterY)
                val closeInTypicalPosition = distance < targetFaceWidth * 1.2
                // also require a roughly similar face size, so we're not matching a close-up
                // shot of one person to a wide shot of someone else who happens to share a
                // screen region
                val similarSizedFraming = candidateFaceWidth / targetFaceWidth in 0.6..1.6

                if (!closeInTypicalPosition || !similarSizedFraming) return@firstOrNull false

                score >= 0.25f
            }

            if (match != null) {
                Log.d(TAG, "POSITION_MERGE candidateSize=${candidate.observations.size} into targetSize=${match.observations.size}")
                match.observations.addAll(candidate.observations)
                match.referenceEmbeddings.addAll(candidate.referenceEmbeddings)
                clusters.remove(candidate)
            }
        }
    }

    private fun cropPerson(
        face: DetectedFaceInstance,
        margin: Float = 1.35f
    ): Bitmap {
        val frame = face.fullFrameBitmap
        val box = face.boundingBox
        val faceWidth = box.width()
        val faceHeight = box.height()

        val cropWidth = (faceWidth * 1.95f).toInt()
        val cropHeight = (faceHeight * 2.70f).toInt()
        val centerX = box.centerX()
        val centerY = box.centerY() + (faceHeight * 0.28f)

        val left = (centerX - cropWidth / 2).coerceIn(0, frame.width)
        val top = (centerY - cropHeight / 2f).toInt().coerceIn(0, frame.height)
        val right = (centerX + cropWidth / 2).coerceIn(left, frame.width)
        val bottom = (centerY + cropHeight / 2f).toInt().coerceIn(top, frame.height)

        return Bitmap.createBitmap(
            frame,
            left,
            top,
            right - left,
            bottom - top
        )
    }


    companion object {
        private const val TAG =
            "PersonClusteringEngine"

        private const val EYE_CLOSED_PENALTY_FACTOR =
            0.1
    }
}