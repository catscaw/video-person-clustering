package com.example.myapp.ml.geometry

import android.graphics.Rect

/** Shrinks [rect] (a proposed crop) just enough that it no longer overlaps any box in
 *  [neighbors], leaving a small [minGapPx] buffer. Shrinks from whichever edge is closest
 *  to each neighbor, so it only gives up as much space as it actually needs to. Used
 *  whenever a crop is widened past a face's own bounding box (for display framing or
 *  embedding padding) so it can never bleed into someone else standing nearby.
 */
fun shrinkToAvoidOverlap(rect: Rect, neighbors: List<Rect>, minGapPx: Int = 4): Rect {
    var left = rect.left
    var top = rect.top
    var right = rect.right
    var bottom = rect.bottom

    val originalWidth = rect.width()
    val originalHeight = rect.height()
    val minAllowedWidth = (originalWidth * 0.5f).toInt()   // never shrink past half size
    val minAllowedHeight = (originalHeight * 0.5f).toInt()

    for (n in neighbors) {
        val overlaps = left < n.right && right > n.left && top < n.bottom && bottom > n.top
        if (!overlaps) continue

        val pushLeft = n.right - left
        val pushRight = right - n.left
        val pushTop = n.bottom - top
        val pushBottom = bottom - n.top
        val minPush = minOf(pushLeft, pushRight, pushTop, pushBottom)

        when (minPush) {
            pushLeft -> if (right - (n.right + minGapPx) >= minAllowedWidth) left = n.right + minGapPx
            pushRight -> if ((n.left - minGapPx) - left >= minAllowedWidth) right = n.left - minGapPx
            pushTop -> if (bottom - (n.bottom + minGapPx) >= minAllowedHeight) top = n.bottom + minGapPx
            else -> if ((n.top - minGapPx) - top >= minAllowedHeight) bottom = n.top - minGapPx
        }
    }

    return Rect(left, top, right.coerceAtLeast(left + 1), bottom.coerceAtLeast(top + 1))
}