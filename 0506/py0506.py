from pathlib import Path
from math import sqrt

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit(
        "This assignment needs Pillow. Install it with: pip install pillow"
    ) from exc


IMAGE_SIZE = (128, 128)
GRADIENT_BINS = 16
MARGIN_C = 0.03


def D(image_path):
    """
    Feature extractor D(x).

    This version uses image differentials. It converts the picture to grayscale,
    computes finite differences dx and dy, then builds a histogram of gradient
    magnitudes. In short, it compares edge/texture changes instead of RGB color.
    """
    image = (
        Image.open(image_path)
        .convert("L")
        .resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    )
    pixels = image.load()
    width, height = image.size

    counts = [0] * GRADIENT_BINS
    total_gradient = 0
    max_gradient = sqrt(255 * 255 + 255 * 255)

    for y in range(height - 1):
        for x in range(width - 1):
            dx = pixels[x + 1, y] - pixels[x, y]
            dy = pixels[x, y + 1] - pixels[x, y]
            gradient = sqrt(dx * dx + dy * dy) / max_gradient
            bin_index = min(int(gradient * GRADIENT_BINS), GRADIENT_BINS - 1)
            counts[bin_index] += 1
            total_gradient += gradient

    sample_count = (width - 1) * (height - 1)
    features = [count / sample_count for count in counts]
    features.append(total_gradient / sample_count)

    length = sqrt(sum(value * value for value in features))
    return [value / length for value in features]


def l2_distance(vector_a, vector_b):
    return sqrt(sum((a - b) ** 2 for a, b in zip(vector_a, vector_b)))


def negative_sample_loss(image_a, image_b, margin=MARGIN_C):
    """
    Assignment 1 negative-sample distance loss:

        L = max(0, C - ||D(x1) - D(x2)||_2)
    """
    distance = l2_distance(D(image_a), D(image_b))
    loss = max(0, margin - distance)
    return distance, loss


def image_distance(image_a, image_b):
    return l2_distance(D(image_a), D(image_b))


def main():
    image_paths = {
        "cat1": Path("cat1.jpg"),
        "cat2": Path("cat2.jpg"),
        "cat4": Path("cat3.jpg"),
        "dog": Path("dog.jpg"),
    }

    missing = [str(path) for path in image_paths.values() if not path.exists()]
    if missing:
        raise SystemExit("Missing image file(s): " + ", ".join(missing))

    pairs = [
        ("cat1", "cat2", "同類參考"),
        ("cat1", "dog", "負樣本"),
        ("cat2", "dog", "負樣本"),
    ]

    #print("作業 1：對比學習 Contrastive Learning")
    print(f"公式：L = max(0, C - ||D(x1) - D(x2)||_2)，C = {MARGIN_C}")
    print()
    print(f"{'圖片組合':<15}{'類型':<12}{'L2 距離':>14}{'負樣本損失':>16}")
    print("-" * 65)

    for left, right, pair_type in pairs:
        distance, loss = negative_sample_loss(image_paths[left], image_paths[right])
        pair_name = left + " vs " + right
        print(f"{pair_name:<15}{pair_type:<12}{distance:>14.6f}{loss:>16.6f}")

    print()
    print("cat4 與其他圖片的距離比較：")
    cat4_distances = []
    for name in ("cat1", "cat2", "dog"):
        distance = image_distance(image_paths["cat4"], image_paths[name])
        cat4_distances.append((name, distance))
        print(f"cat4 vs {name:<5} L2 距離：{distance:.6f}")

    closest_name, closest_distance = min(cat4_distances, key=lambda item: item[1])
    print(f"cat4 最接近的是 {closest_name}，距離為 {closest_distance:.6f}。")
    print()
    print("cat1 vs cat2 只作為同類圖片距離較近的參考。")
    print("對負樣本來說，當距離 >= C 時，損失值會變成 0。")


if __name__ == "__main__":
    main()
