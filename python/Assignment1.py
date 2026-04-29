"""
Assignment 1: Chamfer distance.

This program follows the lecture slide:

    D_chamfer(T, I) = (1 / |T|) * sum_{t in T} d_I(t)

where d_I(t) is the distance from a template feature point to the nearest
feature point in image I.  The distance map is computed by the two-pass
dynamic-programming method shown in the slides.

Usage:
    python Assignment1.py
    python Assignment1.py path/to/image.jpg
    python Assignment1.py path/to/image.jpg --template path/to/template.jpg
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable


INF = 10**9


def import_pillow():
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required for this assignment. Install it with:\n"
            "    pip install pillow"
        ) from exc
    return Image, ImageDraw


def default_image_path() -> Path:
    candidates = [
        Path("images.jpg"),
        Path.cwd() / "images.jpg",
        Path.home() / "Downloads" / "phototest.jpg",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def load_grayscale(path: Path) -> tuple[list[int], int, int]:
    Image, _ = import_pillow()
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
    combined, _, _, threshold = sobel_edge_maps(gray, width, height, threshold)
    return combined, threshold


def sobel_edge_maps(
    gray: list[int],
    width: int,
    height: int,
    threshold: int | None = None,
) -> tuple[list[bool], list[bool], list[bool], int]:
    magnitudes = [0] * (width * height)
    vertical_magnitudes = [0] * (width * height)
    horizontal_magnitudes = [0] * (width * height)

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
            vertical_magnitudes[i] = abs(gx)
            horizontal_magnitudes[i] = abs(gy)

    max_magnitude = max(magnitudes) or 1
    normalized = [int(value * 255 / max_magnitude) for value in magnitudes]
    max_vertical = max(vertical_magnitudes) or 1
    vertical_normalized = [
        int(value * 255 / max_vertical) for value in vertical_magnitudes
    ]
    max_horizontal = max(horizontal_magnitudes) or 1
    horizontal_normalized = [
        int(value * 255 / max_horizontal) for value in horizontal_magnitudes
    ]

    if threshold is None:
        threshold = max(25, otsu_threshold(normalized))

    return (
        [value >= threshold for value in normalized],
        [value >= threshold for value in vertical_normalized],
        [value >= threshold for value in horizontal_normalized],
        threshold,
    )


def distance_transform(
    features: list[bool],
    width: int,
    height: int,
    diagonal: bool = False,
) -> list[float]:
    dist: list[float] = [0.0 if pixel else float(INF) for pixel in features]
    diagonal_weight = math.sqrt(2)

    # Forward pass: closest feature above and to the left.
    for y in range(height):
        for x in range(width):
            i = y * width + x
            best = dist[i]
            if x > 0:
                best = min(best, dist[i - 1] + 1)
            if y > 0:
                best = min(best, dist[i - width] + 1)
            if diagonal:
                if x > 0 and y > 0:
                    best = min(best, dist[i - width - 1] + diagonal_weight)
                if x + 1 < width and y > 0:
                    best = min(best, dist[i - width + 1] + diagonal_weight)
            dist[i] = best

    # Backward pass: closest feature below and to the right.
    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            i = y * width + x
            best = dist[i]
            if x + 1 < width:
                best = min(best, dist[i + 1] + 1)
            if y + 1 < height:
                best = min(best, dist[i + width] + 1)
            if diagonal:
                if x + 1 < width and y + 1 < height:
                    best = min(best, dist[i + width + 1] + diagonal_weight)
                if x > 0 and y + 1 < height:
                    best = min(best, dist[i + width - 1] + diagonal_weight)
            dist[i] = best

    return dist


def feature_points(features: list[bool], width: int, height: int) -> list[tuple[int, int]]:
    return [
        (x, y)
        for y in range(height)
        for x in range(width)
        if features[y * width + x]
    ]


def crop_features(
    features: list[bool],
    width: int,
    x0: int,
    y0: int,
    crop_width: int,
    crop_height: int,
) -> list[bool]:
    cropped = [False] * (crop_width * crop_height)
    for y in range(crop_height):
        source = (y0 + y) * width + x0
        target = y * crop_width
        cropped[target : target + crop_width] = features[source : source + crop_width]
    return cropped


def chamfer_distance(
    template_points: list[tuple[int, int]],
    distance_map: list[float],
    image_width: int,
    image_height: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> float:
    if not template_points:
        raise ValueError("The template has no feature points.")

    total = 0.0
    for x, y in template_points:
        image_x = x + offset_x
        image_y = y + offset_y
        if image_x < 0 or image_y < 0 or image_x >= image_width or image_y >= image_height:
            return float("inf")
        total += distance_map[image_y * image_width + image_x]

    return total / len(template_points)


def best_chamfer_match(
    template_features: list[bool],
    template_width: int,
    template_height: int,
    distance_map: list[float],
    image_width: int,
    image_height: int,
    stride: int = 1,
) -> tuple[float, tuple[int, int]]:
    points = feature_points(template_features, template_width, template_height)
    if not points:
        raise ValueError("The template has no feature points.")

    best_score = float("inf")
    best_offset = (0, 0)

    max_x = image_width - template_width
    max_y = image_height - template_height
    for offset_y in range(0, max_y + 1, stride):
        for offset_x in range(0, max_x + 1, stride):
            total = 0.0
            limit = best_score * len(points)

            for x, y in points:
                total += distance_map[(y + offset_y) * image_width + (x + offset_x)]
                if total >= limit:
                    break

            score = total / len(points)
            if score < best_score:
                best_score = score
                best_offset = (offset_x, offset_y)

    return best_score, best_offset


def save_binary_image(
    features: list[bool],
    width: int,
    height: int,
    output_path: Path,
) -> None:
    Image, _ = import_pillow()
    pixels = [0 if value else 255 for value in features]
    image = Image.new("L", (width, height))
    image.putdata(pixels)
    image.save(output_path)


def save_distance_image(
    distance_map: list[float],
    width: int,
    height: int,
    output_path: Path,
) -> None:
    Image, _ = import_pillow()
    finite = [value for value in distance_map if value < INF]
    max_distance = max(finite) if finite else 1
    if max_distance <= 0:
        max_distance = 1

    pixels = []
    for value in distance_map:
        if value >= INF:
            pixels.append(255)
        else:
            pixels.append(int(255 * value / max_distance))

    image = Image.new("L", (width, height))
    image.putdata(pixels)
    image.save(output_path)


def save_match_overlay(
    image_path: Path,
    box: tuple[int, int, int, int],
    output_path: Path,
) -> None:
    Image, ImageDraw = import_pillow()
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    x, y, width, height = box
    line_width = max(2, min(image.size) // 100)
    draw.rectangle(
        [x, y, x + width - 1, y + height - 1],
        outline=(255, 0, 0),
        width=line_width,
    )
    image.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assignment 1: chamfer distance")
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=default_image_path(),
        help="image whose feature-distance map is computed",
    )
    parser.add_argument(
        "--template",
        type=Path,
        help="optional template image; if omitted, a center crop is used",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="edge threshold from 0 to 255; default uses Otsu's method",
    )
    parser.add_argument(
        "--diagonal",
        action="store_true",
        help="also use diagonal neighbors with sqrt(2) cost",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="matching step size in pixels",
    )
    parser.add_argument(
        "--no-match",
        action="store_true",
        help="only compute the edge map and distance transform",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image

    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    gray, width, height = load_grayscale(image_path)
    image_features, vertical_features, horizontal_features, threshold = sobel_edge_maps(
        gray,
        width,
        height,
        args.threshold,
    )
    distance_map = distance_transform(image_features, width, height, args.diagonal)

    edge_output = Path("assignment1_edges.png")
    vertical_edge_output = Path("assignment1_edges_vertical.png")
    horizontal_edge_output = Path("assignment1_edges_horizontal.png")
    distance_output = Path("assignment1_distance.png")
    save_binary_image(image_features, width, height, edge_output)
    save_binary_image(vertical_features, width, height, vertical_edge_output)
    save_binary_image(horizontal_features, width, height, horizontal_edge_output)
    save_distance_image(distance_map, width, height, distance_output)

    feature_count = sum(image_features)
    print("Assignment 1: chamfer distance")
    print(f"Image: {image_path}")
    print(f"Size: {width} x {height}")
    print(f"Edge threshold: {threshold}")
    print(f"Image feature pixels: {feature_count}")
    print(f"Saved edge map: {edge_output}")
    print(f"Saved vertical edge map: {vertical_edge_output}")
    print(f"Saved horizontal edge map: {horizontal_edge_output}")
    print(f"Saved distance map: {distance_output}")

    if args.no_match:
        return

    if args.template:
        template_gray, template_width, template_height = load_grayscale(args.template)
        template_features, template_threshold = sobel_edges(
            template_gray,
            template_width,
            template_height,
            args.threshold,
        )
        print(f"Template: {args.template}")
        print(f"Template edge threshold: {template_threshold}")
    else:
        template_width = max(8, width // 3)
        template_height = max(8, height // 3)
        crop_x = (width - template_width) // 2
        crop_y = (height - template_height) // 2
        template_features = crop_features(
            image_features,
            width,
            crop_x,
            crop_y,
            template_width,
            template_height,
        )
        print(
            "Template: center crop "
            f"({crop_x}, {crop_y}, {template_width}, {template_height})"
        )

    score, (best_x, best_y) = best_chamfer_match(
        template_features,
        template_width,
        template_height,
        distance_map,
        width,
        height,
        max(1, args.stride),
    )

    overlay_output = Path("assignment1_match.png")
    save_match_overlay(
        image_path,
        (best_x, best_y, template_width, template_height),
        overlay_output,
    )

    print(f"Best match top-left: ({best_x}, {best_y})")
    print(f"Chamfer distance: {score:.4f}")
    print(f"Saved match overlay: {overlay_output}")


if __name__ == "__main__":
    main()
