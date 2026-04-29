"""
Assignment 2: Chi-square distance for shape contexts.

The lecture formula is:

    C_ij = 1/2 * sum_k ((h_i(k) - h_j(k))^2 / (h_i(k) + h_j(k)))

This program builds simple shape-context histograms from image edge points and
uses the chi-square distance to compare point descriptors.

Usage:
    python "Assignment 2.py"
    python "Assignment 2.py" path/to/image.jpg
    python "Assignment 2.py" path/to/image.jpg --other path/to/other.jpg
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, Sequence


Point = tuple[int, int]


def import_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required for this assignment. Install it with:\n"
            "    pip install pillow"
        ) from exc
    return Image, ImageDraw, ImageFont


def default_image_path() -> Path:
    candidates = [
        Path("images.jpg"),
        Path.cwd() / "images.jpg",
        Path.home() / "Downloads" / "images.jpg",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def load_grayscale(path: Path) -> tuple[list[int], int, int]:
    Image, _, _ = import_pillow()
    image = Image.open(path).convert("L")
    width, height = image.size
    if hasattr(image, "get_flattened_data"):
        pixels = image.get_flattened_data()
    else:
        pixels = image.getdata()
    return list(pixels), width, height


def otsu_threshold(values: Iterable[int]) -> int:
    hist = [0] * 256
    total = 0
    for value in values:
        value = max(0, min(255, int(value)))
        hist[value] += 1
        total += 1

    if total == 0:
        return 0

    sum_all = sum(level * count for level, count in enumerate(hist))
    sum_background = 0
    weight_background = 0
    best_variance = -1.0
    best_threshold = 0

    for threshold, count in enumerate(hist):
        weight_background += count
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += threshold * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_all - sum_background) / weight_foreground
        variance = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )

        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold

    return best_threshold


def sobel_edges(
    gray: list[int],
    width: int,
    height: int,
    threshold: int | None = None,
) -> tuple[list[bool], int]:
    magnitudes = [0] * (width * height)

    for y in range(1, height - 1):
        row = y * width
        for x in range(1, width - 1):
            i = row + x
            gx = (
                -gray[i - width - 1]
                + gray[i - width + 1]
                - 2 * gray[i - 1]
                + 2 * gray[i + 1]
                - gray[i + width - 1]
                + gray[i + width + 1]
            )
            gy = (
                gray[i - width - 1]
                + 2 * gray[i - width]
                + gray[i - width + 1]
                - gray[i + width - 1]
                - 2 * gray[i + width]
                - gray[i + width + 1]
            )
            magnitudes[i] = int(math.sqrt(gx * gx + gy * gy))

    max_magnitude = max(magnitudes) or 1
    normalized = [int(value * 255 / max_magnitude) for value in magnitudes]

    if threshold is None:
        threshold = max(25, otsu_threshold(normalized))

    return [value >= threshold for value in normalized], threshold


def edge_points(features: list[bool], width: int, height: int) -> list[Point]:
    return [
        (x, y)
        for y in range(height)
        for x in range(width)
        if features[y * width + x]
    ]


def sample_points(points: Sequence[Point], max_points: int) -> list[Point]:
    if max_points <= 0:
        raise ValueError("max_points must be positive.")
    if len(points) <= max_points:
        return list(points)
    if max_points == 1:
        return [points[len(points) // 2]]

    sampled: list[Point] = []
    last_index = len(points) - 1
    for i in range(max_points):
        index = round(i * last_index / (max_points - 1))
        sampled.append(points[index])
    return sampled


def mean_pairwise_distance(points: Sequence[Point]) -> float:
    if len(points) < 2:
        return 1.0

    total = 0.0
    count = 0
    for i, (x1, y1) in enumerate(points):
        for x2, y2 in points[i + 1 :]:
            total += math.hypot(x2 - x1, y2 - y1)
            count += 1

    if count == 0 or total <= 0:
        return 1.0
    return total / count


def log_space(start: float, stop: float, count: int) -> list[float]:
    if start <= 0 or stop <= 0:
        raise ValueError("log-space values must be positive.")
    if count < 2:
        raise ValueError("count must be at least 2.")

    log_start = math.log(start)
    log_stop = math.log(stop)
    return [
        math.exp(log_start + (log_stop - log_start) * i / (count - 1))
        for i in range(count)
    ]


def radial_bin_index(distance: float, radial_edges: Sequence[float]) -> int | None:
    if distance < radial_edges[0] or distance > radial_edges[-1]:
        return None

    for i in range(len(radial_edges) - 1):
        if radial_edges[i] <= distance < radial_edges[i + 1]:
            return i

    return len(radial_edges) - 2


def shape_context_histograms(
    points: Sequence[Point],
    radial_bins: int = 5,
    angular_bins: int = 12,
    inner_radius: float = 0.125,
    outer_radius: float = 2.0,
    normalize: bool = True,
) -> list[list[float]]:
    if radial_bins <= 0 or angular_bins <= 0:
        raise ValueError("radial_bins and angular_bins must be positive.")

    mean_distance = mean_pairwise_distance(points)
    radial_edges = log_space(inner_radius, outer_radius, radial_bins + 1)
    bin_count = radial_bins * angular_bins
    histograms: list[list[float]] = []

    for i, (center_x, center_y) in enumerate(points):
        histogram = [0.0] * bin_count

        for j, (x, y) in enumerate(points):
            if i == j:
                continue

            dx = x - center_x
            dy = y - center_y
            normalized_distance = math.hypot(dx, dy) / mean_distance
            r_bin = radial_bin_index(normalized_distance, radial_edges)
            if r_bin is None:
                continue

            angle = math.atan2(dy, dx)
            if angle < 0:
                angle += 2 * math.pi
            theta_bin = min(
                angular_bins - 1,
                int(angle / (2 * math.pi) * angular_bins),
            )
            histogram[r_bin * angular_bins + theta_bin] += 1.0

        if normalize:
            total = sum(histogram)
            if total > 0:
                histogram = [value / total for value in histogram]

        histograms.append(histogram)

    return histograms


def chi_square_distance(hist_i: Sequence[float], hist_j: Sequence[float]) -> float:
    if len(hist_i) != len(hist_j):
        raise ValueError("Histograms must have the same length.")

    total = 0.0
    for value_i, value_j in zip(hist_i, hist_j):
        denominator = value_i + value_j
        if denominator == 0:
            continue
        total += (value_i - value_j) ** 2 / denominator

    return 0.5 * total


def chi_square_formula_terms(
    hist_i: Sequence[float],
    hist_j: Sequence[float],
) -> list[tuple[int, float, float, float, float, float]]:
    if len(hist_i) != len(hist_j):
        raise ValueError("Histograms must have the same length.")

    terms = []
    for bin_index, (value_i, value_j) in enumerate(zip(hist_i, hist_j), start=1):
        numerator = (value_i - value_j) ** 2
        denominator = value_i + value_j
        term = 0.0
        if denominator != 0:
            term = 0.5 * numerator / denominator
        terms.append((bin_index, value_i, value_j, numerator, denominator, term))
    return terms


def nearest_to_center(points: Sequence[Point], width: int, height: int) -> int:
    center_x = width / 2
    center_y = height / 2
    return min(
        range(len(points)),
        key=lambda i: (points[i][0] - center_x) ** 2 + (points[i][1] - center_y) ** 2,
    )


def ranked_distances(
    histograms: Sequence[Sequence[float]],
    reference_index: int,
) -> list[tuple[float, int]]:
    reference = histograms[reference_index]
    distances = [
        (chi_square_distance(reference, histogram), index)
        for index, histogram in enumerate(histograms)
        if index != reference_index
    ]
    distances.sort(key=lambda item: item[0])
    return distances


def best_cross_image_matches(
    source_histograms: Sequence[Sequence[float]],
    target_histograms: Sequence[Sequence[float]],
) -> list[tuple[float, int, int]]:
    matches: list[tuple[float, int, int]] = []

    for source_index, source_histogram in enumerate(source_histograms):
        best_distance = float("inf")
        best_target_index = 0

        for target_index, target_histogram in enumerate(target_histograms):
            distance = chi_square_distance(source_histogram, target_histogram)
            if distance < best_distance:
                best_distance = distance
                best_target_index = target_index

        matches.append((best_distance, source_index, best_target_index))

    matches.sort(key=lambda item: item[0])
    return matches


def save_binary_image(
    features: list[bool],
    width: int,
    height: int,
    output_path: Path,
) -> None:
    Image, _, _ = import_pillow()
    pixels = [0 if value else 255 for value in features]
    image = Image.new("L", (width, height))
    image.putdata(pixels)
    image.save(output_path)


def save_point_overlay(
    image_path: Path,
    points: Sequence[Point],
    output_path: Path,
    reference_index: int | None = None,
    similar_index: int | None = None,
    different_index: int | None = None,
) -> None:
    Image, ImageDraw, _ = import_pillow()
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    radius = max(2, min(image.size) // 90)

    for x, y in points:
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(0, 180, 255),
            outline=(255, 255, 255),
        )

    highlights = [
        (reference_index, (255, 40, 40)),
        (similar_index, (0, 190, 75)),
        (different_index, (190, 60, 255)),
    ]
    for index, color in highlights:
        if index is None:
            continue
        x, y = points[index]
        draw.ellipse(
            [x - radius * 2, y - radius * 2, x + radius * 2, y + radius * 2],
            fill=color,
            outline=(255, 255, 255),
            width=2,
        )

    image.save(output_path)


def heat_color(value: float) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, value))
    if value < 0.5:
        t = value / 0.5
        red = int(255 * t)
        green = int(220 * t)
        blue = 255 - int(185 * t)
    else:
        t = (value - 0.5) / 0.5
        red = 255
        green = 220 - int(170 * t)
        blue = 70 - int(70 * t)
    return red, green, blue


def save_histogram_image(
    histogram: Sequence[float],
    radial_bins: int,
    angular_bins: int,
    output_path: Path,
) -> None:
    Image, ImageDraw, ImageFont = import_pillow()
    cell = 34
    label_space = 44
    width = label_space + angular_bins * cell
    height = label_space + radial_bins * cell
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    max_value = max(histogram) if histogram else 0
    if max_value <= 0:
        max_value = 1

    for r in range(radial_bins):
        y0 = label_space + r * cell
        draw.text((8, y0 + 10), f"r{r + 1}", fill=(35, 35, 35), font=font)
        for a in range(angular_bins):
            x0 = label_space + a * cell
            value = histogram[r * angular_bins + a]
            color = heat_color(value / max_value)
            draw.rectangle(
                [x0, y0, x0 + cell - 1, y0 + cell - 1],
                fill=color,
                outline=(220, 220, 220),
            )

    for a in range(angular_bins):
        x0 = label_space + a * cell
        draw.text((x0 + 8, 16), str(a + 1), fill=(35, 35, 35), font=font)

    draw.text((8, 16), "bin", fill=(35, 35, 35), font=font)
    image.save(output_path)


def save_correspondence_overlay(
    source_image_path: Path,
    target_image_path: Path,
    source_points: Sequence[Point],
    target_points: Sequence[Point],
    matches: Sequence[tuple[float, int, int]],
    output_path: Path,
    max_lines: int,
) -> None:
    Image, ImageDraw, _ = import_pillow()
    source_image = Image.open(source_image_path).convert("RGB")
    target_image = Image.open(target_image_path).convert("RGB")
    gap = 24
    width = source_image.width + gap + target_image.width
    height = max(source_image.height, target_image.height)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(source_image, (0, 0))
    canvas.paste(target_image, (source_image.width + gap, 0))
    draw = ImageDraw.Draw(canvas)
    radius = max(2, min(source_image.size + target_image.size) // 160)

    for distance, source_index, target_index in matches[:max_lines]:
        del distance
        source_x, source_y = source_points[source_index]
        target_x, target_y = target_points[target_index]
        target_x += source_image.width + gap
        draw.line([source_x, source_y, target_x, target_y], fill=(255, 170, 0), width=1)
        draw.ellipse(
            [source_x - radius, source_y - radius, source_x + radius, source_y + radius],
            fill=(255, 40, 40),
            outline=(255, 255, 255),
        )
        draw.ellipse(
            [target_x - radius, target_y - radius, target_x + radius, target_y + radius],
            fill=(0, 190, 75),
            outline=(255, 255, 255),
        )

    canvas.save(output_path)


def save_distances_csv(
    distances: Sequence[tuple[float, int]],
    points: Sequence[Point],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["rank", "point_index", "x", "y", "chi_square_distance"])
        for rank, (distance, index) in enumerate(distances, start=1):
            x, y = points[index]
            writer.writerow([rank, index, x, y, f"{distance:.8f}"])


def save_formula_terms_csv(
    reference_histogram: Sequence[float],
    pairs: Sequence[tuple[str, int, Point, Sequence[float]]],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "pair",
                "point_index",
                "x",
                "y",
                "k",
                "h_i(k)",
                "h_j(k)",
                "numerator",
                "denominator",
                "0.5*numerator/denominator",
            ]
        )

        for pair_name, point_index, (x, y), histogram in pairs:
            for (
                bin_index,
                value_i,
                value_j,
                numerator,
                denominator,
                term,
            ) in chi_square_formula_terms(reference_histogram, histogram):
                writer.writerow(
                    [
                        pair_name,
                        point_index,
                        x,
                        y,
                        bin_index,
                        f"{value_i:.8f}",
                        f"{value_j:.8f}",
                        f"{numerator:.8f}",
                        f"{denominator:.8f}",
                        f"{term:.8f}",
                    ]
                )


def process_image(
    image_path: Path,
    threshold: int | None,
    max_points: int,
    radial_bins: int,
    angular_bins: int,
    inner_radius: float,
    outer_radius: float,
) -> tuple[list[bool], int, int, list[Point], list[list[float]], int]:
    gray, width, height = load_grayscale(image_path)
    features, threshold_used = sobel_edges(gray, width, height, threshold)
    points = sample_points(edge_points(features, width, height), max_points)

    if len(points) < 2:
        raise SystemExit(
            "Not enough edge points were found. Try a lower --threshold value."
        )

    histograms = shape_context_histograms(
        points,
        radial_bins=radial_bins,
        angular_bins=angular_bins,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
    )
    return features, width, height, points, histograms, threshold_used


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assignment 2: chi-square distance for shape contexts"
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=default_image_path(),
        help="input image used to build shape-context histograms",
    )
    parser.add_argument(
        "--other",
        type=Path,
        help="optional second image for cross-image descriptor matching",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="edge threshold from 0 to 255; default uses Otsu's method",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=160,
        help="maximum number of edge points sampled for shape contexts",
    )
    parser.add_argument(
        "--radial-bins",
        type=int,
        default=5,
        help="number of log-distance bins",
    )
    parser.add_argument(
        "--angular-bins",
        type=int,
        default=12,
        help="number of angle bins",
    )
    parser.add_argument(
        "--inner-radius",
        type=float,
        default=0.125,
        help="smallest normalized log-radius bin boundary",
    )
    parser.add_argument(
        "--outer-radius",
        type=float,
        default=2.0,
        help="largest normalized log-radius bin boundary",
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        default=None,
        help="sampled point index used as h_i; default is closest to image center",
    )
    parser.add_argument(
        "--matches",
        type=int,
        default=20,
        help="number of best cross-image matches drawn when --other is used",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image

    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    (
        features,
        width,
        height,
        points,
        histograms,
        threshold_used,
    ) = process_image(
        image_path,
        args.threshold,
        args.points,
        args.radial_bins,
        args.angular_bins,
        args.inner_radius,
        args.outer_radius,
    )

    reference_index = args.reference_index
    if reference_index is None:
        reference_index = nearest_to_center(points, width, height)
    if reference_index < 0 or reference_index >= len(points):
        raise SystemExit(
            f"--reference-index must be between 0 and {len(points) - 1}."
        )

    distances = ranked_distances(histograms, reference_index)
    similar_distance, similar_index = distances[0]
    different_distance, different_index = distances[-1]

    edge_output = Path("assignment2_edges.png")
    points_output = Path("assignment2_points.png")
    histogram_output = Path("assignment2_reference_histogram.png")
    csv_output = Path("assignment2_distances.csv")
    formula_output = Path("assignment2_formula_terms.csv")
    save_binary_image(features, width, height, edge_output)
    save_point_overlay(
        image_path,
        points,
        points_output,
        reference_index=reference_index,
        similar_index=similar_index,
        different_index=different_index,
    )
    save_histogram_image(
        histograms[reference_index],
        args.radial_bins,
        args.angular_bins,
        histogram_output,
    )
    save_distances_csv(distances, points, csv_output)
    save_formula_terms_csv(
        histograms[reference_index],
        [
            ("most_similar", similar_index, points[similar_index], histograms[similar_index]),
            (
                "most_different",
                different_index,
                points[different_index],
                histograms[different_index],
            ),
        ],
        formula_output,
    )

    print("Assignment 2: chi-square distance")
    print(f"Image: {image_path}")
    print(f"Size: {width} x {height}")
    print(f"Edge threshold: {threshold_used}")
    print(f"Sampled edge points: {len(points)}")
    print(f"Histogram bins K: {args.radial_bins * args.angular_bins}")
    print(f"Reference point index: {reference_index}, point: {points[reference_index]}")
    print(
        "Most similar point: "
        f"index {similar_index}, point {points[similar_index]}, "
        f"C_ij = {similar_distance:.6f}"
    )
    print(
        "Most different point: "
        f"index {different_index}, point {points[different_index]}, "
        f"C_ij = {different_distance:.6f}"
    )
    print(f"Saved edge map: {edge_output}")
    print(f"Saved point overlay: {points_output}")
    print(f"Saved reference histogram: {histogram_output}")
    print(f"Saved chi-square ranking CSV: {csv_output}")
    print(f"Saved chi-square formula terms CSV: {formula_output}")

    if args.other:
        other_path = args.other
        if not other_path.exists():
            raise SystemExit(f"Second image not found: {other_path}")

        (
            other_features,
            other_width,
            other_height,
            other_points,
            other_histograms,
            other_threshold,
        ) = process_image(
            other_path,
            args.threshold,
            args.points,
            args.radial_bins,
            args.angular_bins,
            args.inner_radius,
            args.outer_radius,
        )

        del other_features, other_width, other_height, other_threshold
        matches = best_cross_image_matches(histograms, other_histograms)
        match_output = Path("assignment2_correspondences.png")
        save_correspondence_overlay(
            image_path,
            other_path,
            points,
            other_points,
            matches,
            match_output,
            max(1, args.matches),
        )
        average_best = sum(distance for distance, _, _ in matches) / len(matches)
        best_distance, source_index, target_index = matches[0]
        print(f"Second image: {other_path}")
        print(f"Second image sampled edge points: {len(other_points)}")
        print(f"Best cross-image match cost: {best_distance:.6f}")
        print(
            "Best cross-image pair: "
            f"source {source_index} {points[source_index]} -> "
            f"target {target_index} {other_points[target_index]}"
        )
        print(f"Average best-match cost: {average_best:.6f}")
        print(f"Saved correspondence overlay: {match_output}")


if __name__ == "__main__":
    main()
