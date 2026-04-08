import argparse
from collections import Counter, deque
from math import sqrt
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw


class KNNClassifier:
    def __init__(self, k=3):
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self.k = k
        self.x_train = []
        self.y_train = []

    def fit(self, x_values, y_values):
        if len(x_values) != len(y_values):
            raise ValueError("x_values and y_values must have the same length")
        if len(x_values) < self.k:
            raise ValueError("number of training samples must be >= k")
        self.x_train = x_values
        self.y_train = y_values

    @staticmethod
    def _euclidean_distance(point_a, point_b):
        return sqrt(sum((a - b) ** 2 for a, b in zip(point_a, point_b)))

    def predict_one(self, point):
        distances = []
        for train_point, label in zip(self.x_train, self.y_train):
            distance = self._euclidean_distance(point, train_point)
            distances.append((distance, label))

        distances.sort(key=lambda item: item[0])
        nearest_labels = [label for _, label in distances[: self.k]]
        return Counter(nearest_labels).most_common(1)[0][0]


LABEL_STYLES = {
    "orange": {"fill": (255, 153, 51, 95), "point": "#ff8c2a"},
    "green": {"fill": (74, 179, 84, 95), "point": "#34a853"},
    "blue": {"fill": (52, 152, 219, 95), "point": "#2196f3"},
}

IMAGE_PATH = Path("image.jpg")
OUTPUT_PREFIX = "knn_result"
MIN_COMPONENT_SIZE = 12


def classify_color(pixel):
    red, green, blue = pixel

    if red > 180 and 70 < green < 210 and blue < 140:
        return "orange"
    if green > 120 and red < 140 and blue < 170:
        return "green"
    if blue > 140 and green > 80 and red < 140:
        return "blue"
    return None


def find_points(image):
    width, height = image.size
    visited = [[False for _ in range(width)] for _ in range(height)]
    pixels = image.load()
    points = []

    for y in range(height):
        for x in range(width):
            if visited[y][x]:
                continue

            label = classify_color(pixels[x, y])
            if label is None:
                continue

            queue = deque([(x, y)])
            visited[y][x] = True
            component = []

            while queue:
                current_x, current_y = queue.popleft()
                component.append((current_x, current_y))

                for diff_x, diff_y in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                ):
                    next_x = current_x + diff_x
                    next_y = current_y + diff_y

                    if not (0 <= next_x < width and 0 <= next_y < height):
                        continue
                    if visited[next_y][next_x]:
                        continue

                    next_label = classify_color(pixels[next_x, next_y])
                    if next_label == label:
                        visited[next_y][next_x] = True
                        queue.append((next_x, next_y))

            if len(component) >= MIN_COMPONENT_SIZE:
                average_x = sum(point_x for point_x, _ in component) / len(component)
                average_y = sum(point_y for _, point_y in component) / len(component)
                points.append(
                    {
                        "label": label,
                        "x": average_x,
                        "y": average_y,
                        "size": len(component),
                    }
                )

    return sorted(points, key=lambda item: (item["label"], item["y"], item["x"]))


def build_prediction_map(model, size):
    width, height = size
    predictions = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(model.predict_one((x, y)))
        predictions.append(row)
    return predictions


def draw_result(base_image, points, predictions, output_path):
    width, height = base_image.size
    overlay = Image.new("RGBA", base_image.size, (255, 255, 255, 0))
    overlay_pixels = overlay.load()

    for y in range(height):
        for x in range(width):
            overlay_pixels[x, y] = LABEL_STYLES[predictions[y][x]]["fill"]

    blended = Image.blend(base_image.convert("RGBA"), overlay, alpha=0.42)
    draw = ImageDraw.Draw(blended)

    for y in range(height - 1):
        for x in range(width - 1):
            current_label = predictions[y][x]
            if predictions[y][x + 1] != current_label or predictions[y + 1][x] != current_label:
                draw.point((x, y), fill=(0, 0, 0, 255))

    for point in points:
        x = point["x"]
        y = point["y"]
        color = ImageColor.getrgb(LABEL_STYLES[point["label"]]["point"])
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(0, 0, 0, 255), width=2)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)

    blended.save(output_path)


def resolve_image_path():
    if IMAGE_PATH.exists():
        return IMAGE_PATH

    jpg_files = sorted(Path(".").glob("*.jpg"))
    if not jpg_files:
        raise FileNotFoundError("No .jpg image found in the current folder")
    return jpg_files[0]


def resolve_k_value():
    parser = argparse.ArgumentParser(description="Draw KNN decision regions from an image.")
    parser.add_argument("-k", "--neighbors", type=int, help="Number of neighbors for KNN")
    args = parser.parse_args()

    if args.neighbors is not None:
        if args.neighbors <= 0:
            raise ValueError("k must be greater than 0")
        return args.neighbors

    raw_value = input("請輸入 k 值 (直接按 Enter 使用 3): ").strip()
    if not raw_value:
        return 3
    k_value = int(raw_value)
    if k_value <= 0:
        raise ValueError("k must be greater than 0")
    return k_value


def build_output_path(k_value):
    return Path(f"{OUTPUT_PREFIX}_k{k_value}.png")


def main():
    k_value = resolve_k_value()
    image_path = resolve_image_path()
    image = Image.open(image_path).convert("RGB")
    points = find_points(image)

    train_x = [(point["x"], point["y"]) for point in points]
    train_y = [point["label"] for point in points]

    model = KNNClassifier(k=k_value)
    model.fit(train_x, train_y)

    predictions = build_prediction_map(model, image.size)
    output_path = build_output_path(k_value)
    draw_result(image, points, predictions, output_path)

    print(f"Using k = {k_value}")
    print(f"Detected {len(points)} points from {image_path}")
    for point in points:
        print(
            f"{point['label']:>6} -> ({point['x']:.2f}, {point['y']:.2f}) "
            f"pixels={point['size']}"
        )
    print(f"Saved result to {output_path}")


if __name__ == "__main__":
    main()
