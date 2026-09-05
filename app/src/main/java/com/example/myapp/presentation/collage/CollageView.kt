package com.example.myapp.presentation.collage

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.myapp.ml.clustering.PersonResult
import kotlin.math.ceil
import kotlin.math.sqrt

/**
 * Renders [people] as a scrapbook-style mosaic of overlapping "polaroids" — the popular
 * Instagram Story collage look — rather than a plain photo grid. Layout changes based on the
 * number of unique identities so a single person fills the frame, two people split evenly,
 * and larger groups fall back to a uniform grid capped at [maxTilesBeforeOverflow] with a
 * "+N more" tile for anyone who doesn't fit.
 *
 * Crop expectation: [PersonResult.bestFrameBitmap] is a generous head-and-shoulders crop
 * (see `PersonClusteringEngine.cropGenerously`), not the tight ML Kit face bounding box —
 * so it holds up reasonably well when stretched to fill a tile via `ContentScale.Crop`.
 */
@Composable
fun CollageView(
    people: List<PersonResult>,
    modifier: Modifier = Modifier,
    maxTilesBeforeOverflow: Int = 9,
    title: String = "the cast"
) {
    val sortedByPresence = remember(people) { people.sortedByDescending { it.totalAppearances } }

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(
                Brush.linearGradient(
                    colors = listOf(Color(0xFFF5C4B3), Color(0xFFF4C0D1), Color(0xFFCECBF6))
                )
            )
    ) {
        Column(Modifier.fillMaxSize().padding(20.dp)) {
            if (sortedByPresence.isNotEmpty()) {
                CollageHeader(title = title, peopleCount = sortedByPresence.size)
                Spacer(Modifier.height(10.dp))
            }

            Box(Modifier.weight(1f).fillMaxWidth()) {
                when (sortedByPresence.size) {
                    0 -> EmptyCollagePlaceholder()
                    1 -> PersonTile(
                        person = sortedByPresence[0],
                        rank = 0,
                        modifier = Modifier.fillMaxSize().padding(6.dp),
                        rotateForEffect = false
                    )
                    2 -> Row(Modifier.fillMaxSize()) {
                        sortedByPresence.forEachIndexed { index, person ->
                            PersonTile(person, index, Modifier.weight(1f).fillMaxHeight().padding(10.dp))
                        }
                    }
                    3 -> Row(Modifier.fillMaxSize()) {
                        // One larger "hero" tile + two stacked smaller tiles, mirrors common
                        // story-collage templates rather than a plain even 3-way split.
                        PersonTile(
                            sortedByPresence[0],
                            0,
                            Modifier.weight(1.4f).fillMaxHeight().padding(10.dp)
                        )
                        Column(Modifier.weight(1f).fillMaxHeight()) {
                            PersonTile(sortedByPresence[1], 1, Modifier.weight(1f).fillMaxWidth().padding(10.dp))
                            PersonTile(sortedByPresence[2], 2, Modifier.weight(1f).fillMaxWidth().padding(10.dp))
                        }
                    }
                    4 -> Column(Modifier.fillMaxSize()) {
                        Row(Modifier.weight(1f).fillMaxWidth()) {
                            PersonTile(sortedByPresence[0], 0, Modifier.weight(1f).fillMaxHeight().padding(10.dp))
                            PersonTile(sortedByPresence[1], 1, Modifier.weight(1f).fillMaxHeight().padding(10.dp))
                        }
                        Row(Modifier.weight(1f).fillMaxWidth()) {
                            PersonTile(sortedByPresence[2], 2, Modifier.weight(1f).fillMaxHeight().padding(10.dp))
                            PersonTile(sortedByPresence[3], 3, Modifier.weight(1f).fillMaxHeight().padding(10.dp))
                        }
                    }
                    else -> {
                        val (visible, overflowCount) = if (sortedByPresence.size > maxTilesBeforeOverflow) {
                            // Reserve the last slot for a "+N more" tile.
                            sortedByPresence.take(maxTilesBeforeOverflow - 1) to
                                    (sortedByPresence.size - (maxTilesBeforeOverflow - 1))
                        } else {
                            sortedByPresence to 0
                        }
                        val columns = ceil(sqrt(visible.size.toDouble() + if (overflowCount > 0) 1 else 0)).toInt()
                            .coerceAtLeast(2)

                        UniformGrid(
                            people = visible,
                            columns = columns,
                            overflowCount = overflowCount,
                            modifier = Modifier.fillMaxSize()
                        )
                    }
                }
            }
        }

        if (sortedByPresence.isNotEmpty()) {
            Text(
                text = "made with video collage",
                color = Color(0xFF4A1B0C).copy(alpha = 0.65f),
                fontSize = 11.sp,
                fontWeight = FontWeight.Medium,
                letterSpacing = 0.5.sp,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 14.dp)
                    .clip(RoundedCornerShape(percent = 50))
                    .background(Color.White.copy(alpha = 0.55f))
                    .padding(horizontal = 14.dp, vertical = 6.dp)
            )
        }
    }
}

@Composable
private fun CollageHeader(title: String, peopleCount: Int) {
    Column {
        Text(
            text = title,
            color = Color(0xFF4A1B0C),
            fontSize = 30.sp,
            fontStyle = FontStyle.Italic,
            fontWeight = FontWeight.Medium
        )
        Text(
            text = "$peopleCount ${if (peopleCount == 1) "person" else "people"} found",
            color = Color(0xFF712B13).copy(alpha = 0.75f),
            fontSize = 12.sp,
            fontWeight = FontWeight.Medium
        )
    }
}

/** Non-lazy, evenly weighted grid so every tile is guaranteed to actually compose and draw —
 * important because [CollageExporter] renders this off-screen, where a Lazy layout would only
 * compose items currently within its (nonexistent) viewport. */
@Composable
private fun UniformGrid(
    people: List<PersonResult>,
    columns: Int,
    overflowCount: Int,
    modifier: Modifier = Modifier
) {
    val rows = people.chunked(columns)

    Column(modifier = modifier) {
        rows.forEachIndexed { rowIndex, rowItems ->
            Row(Modifier.weight(1f).fillMaxWidth()) {
                rowItems.forEachIndexed { columnIndex, person ->
                    val rank = rowIndex * columns + columnIndex
                    PersonTile(person, rank, Modifier.weight(1f).fillMaxHeight().padding(8.dp))
                }
                val isLastRow = rowIndex == rows.lastIndex
                val emptySlots = columns - rowItems.size - if (isLastRow && overflowCount > 0) 1 else 0
                if (isLastRow && overflowCount > 0) {
                    OverflowTile(overflowCount, Modifier.weight(1f).fillMaxHeight().padding(8.dp))
                }
                repeat(emptySlots.coerceAtLeast(0)) {
                    Spacer(Modifier.weight(1f))
                }
            }
        }
    }
}

/** Small alternating tilt per tile so the grid reads as scattered polaroids rather than a
 * rigid layout — deliberately subtle angles only, never enough to threaten legibility or spill
 * badly outside the tile's own slot. */
private fun tiltForRank(rank: Int): Float = when (rank % 4) {
    0 -> -4f
    1 -> 3f
    2 -> -3f
    else -> 4f
}

@Composable
private fun PersonTile(
    person: PersonResult,
    rank: Int,
    modifier: Modifier = Modifier,
    rotateForEffect: Boolean = true
) {
    val shape = RoundedCornerShape(4.dp)
    val rotation = if (rotateForEffect) tiltForRank(rank) else 0f

    Column(
        modifier = modifier
            .graphicsLayer { rotationZ = rotation }
            .shadow(elevation = 8.dp, shape = shape, clip = false)
            .clip(shape)
            .background(Color.White)
            .border(1.dp, Color.White, shape)
    ) {
        Image(
            bitmap = person.bestFrameBitmap.asImageBitmap(),
            contentDescription = null,
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .padding(if (rotateForEffect) 5.dp else 0.dp)
                .clip(RoundedCornerShape(1.dp)),
            contentScale = ContentScale.Crop
        )

        if (rotateForEffect) {
            Text(
                text = "${person.totalAppearances} appearance${if (person.totalAppearances == 1) "" else "s"}",
                color = Color(0xFF2C2C2A),
                fontSize = 10.sp,
                fontWeight = FontWeight.Medium,
                modifier = Modifier.padding(start = 7.dp, top = 2.dp, bottom = 7.dp)
            )
        }
    }
}

@Composable
private fun OverflowTile(overflowCount: Int, modifier: Modifier = Modifier) {
    val shape = RoundedCornerShape(4.dp)
    Column(
        modifier = modifier
            .graphicsLayer { rotationZ = -3f }
            .shadow(elevation = 8.dp, shape = shape, clip = false)
            .clip(shape)
            .background(Color.White)
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .padding(5.dp)
                .clip(RoundedCornerShape(1.dp))
                .background(Color(0xFFEEEDFE)),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = "+$overflowCount",
                    color = Color(0xFF3C3489),
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "more",
                    color = Color(0xFF3C3489).copy(alpha = 0.7f),
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Medium
                )
            }
        }
        Spacer(Modifier.height(7.dp))
    }
}

@Composable
private fun EmptyCollagePlaceholder(modifier: Modifier = Modifier) {
    Box(modifier = modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = "No people detected",
                color = Color(0xFF4A1B0C),
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium,
                textAlign = TextAlign.Center
            )
            Spacer(Modifier.height(4.dp))
            Text(
                text = "Try a video with clearer faces",
                color = Color(0xFF4A1B0C).copy(alpha = 0.6f),
                fontSize = 12.sp,
                textAlign = TextAlign.Center
            )
        }
    }
}