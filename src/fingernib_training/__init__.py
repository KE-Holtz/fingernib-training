import csv
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

from yaml.reader import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

Point = dict[str, float]  # Every point has an x, y, and t.
Stroke = list[Point]  # One pen up to pen down gesture
Strokes = list[Stroke]  # One character worth of strokes
# (There can be more than one for letters like i, j, t)
NormalizedStrokes = list[NDArray[np.float64]]  # each array has shape (n_points, 2)

# Parsing results
RawRow = tuple[int, str, str]
Result = tuple[int, str, NDArray[np.float32]]


class StrokeImages(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, images: NDArray[np.float32], targets: NDArray[np.int64]) -> None:
        # (N, 32, 32) -> (N, 1, 32, 32): CNNs expect a channel dimension
        self.images = torch.from_numpy(images).unsqueeze(1)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.targets[index]


class Embedder(nn.Module):
    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(1, 32),
            self._block(32, 64),
            self._block(64, 128),
        )
        self.project = nn.Linear(128 * 4 * 4, embedding_dim)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.project(x)


class Classifier(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.embedder = Embedder(embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedder(x))


# If true, generate the cache and do nothing else
GENERATE_CACHE = False
# If true, train the model and do nothing else
TRAIN = False
# If true, run diagnostics on the model and do nothing else
TEST = True
LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

# populate the symbol_id_map with IDs and their coresponding symbol or latex
csv.field_size_limit(10**9)


def main() -> None:
    if GENERATE_CACHE:
        build_cache("data/train-data.csv", "cache/train-data-cache.npz")
        build_cache("data/test-data.csv", "cache/test-data-cache.npz")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbol_id_map = generate_id_map("data/symbols.csv")
    train_data = np.load("cache/train-data-cache.npz")
    images = train_data["images"]
    labels = train_data["labels"]
    class_ids = np.array(sorted(symbol_id_map), dtype=np.int64)
    id_to_index = {sid: i for i, sid in enumerate(class_ids.tolist())}

    targets = np.array([id_to_index[sid] for sid in labels.tolist()], dtype=np.int64)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(targets))
    n_val = len(order) // 10
    val_idx, train_idx = order[:n_val], order[n_val:]

    train_set = StrokeImages(images[train_idx], targets[train_idx].astype(np.int64))
    val_set = StrokeImages(images[val_idx], targets[val_idx].astype(np.int64))
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256)
    batch_images, batch_targets = next(iter(train_loader))

    if TRAIN:
        model = Classifier(num_classes=len(class_ids)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        EPOCHS = 10
        best_val_acc = 0.0

        model = Classifier(num_classes=len(class_ids)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, device
            )
            val_loss, val_acc = evaluate(model, val_loader, device)

            print(
                f"epoch {epoch:2d}  "
                f"train loss {train_loss:.4f} acc {train_acc:.1%}  |  "
                f"val loss {val_loss:.4f} acc {val_acc:.1%}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {"model": model.state_dict(), "class_ids": class_ids},
                    "cache/classifier.pt",
                )
        return

    if TEST:
        ckpt = torch.load(
            "cache/classifier.pt", map_location=device, weights_only=False
        )
        model = Classifier(num_classes=len(class_ids)).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        logits, targets_all = predict_logits(model, val_loader, device)

        # 1. Top-k accuracy: is the right answer among the model's k best guesses?
        top5 = logits.topk(5, dim=1).indices
        hits = top5 == targets_all[:, None]
        for k in (1, 2, 3, 4, 5):
            print(f"top-{k}: {hits[:, :k].any(dim=1).float().mean().item():.1%}")

        # 2. The most common mistakes, named
        preds = logits.argmax(dim=1)
        wrong = preds != targets_all
        pairs = Counter(zip(targets_all[wrong].tolist(), preds[wrong].tolist()))
        for (true_idx, pred_idx), count in pairs.most_common(25):
            true_name = symbol_id_map[int(class_ids[true_idx])]
            pred_name = symbol_id_map[int(class_ids[pred_idx])]
            print(f"{count:4d}  {true_name}  ->  {pred_name}")

        alnum_accuracy(logits, targets_all, class_ids, symbol_id_map)
        return


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_logits, all_targets = [], []
    for images, targets in loader:
        all_logits.append(model(images.to(device)).cpu())
        all_targets.append(targets)
    return torch.cat(all_logits), torch.cat(all_targets)


def generate_id_map(path: str | Path) -> dict[int, str]:
    symbol_id_map = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter=";", quotechar="'"):
            symbol_id = int(
                row["symbol_id"],
            )
            symbol = row["latex"]
            symbol_id_map[symbol_id] = symbol

    return symbol_id_map


def load(path: str | Path) -> Iterator[tuple[int, str, Strokes]]:
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter=";", quotechar="'"):
            yield (
                int(row["symbol_id"]),
                row["user_id"],
                yaml.load(row["data"], LOADER),
            )


def normalize(strokes: Strokes) -> NormalizedStrokes:
    # Since the raw X and Y of the data is likely irrelevant to what the actual character is, normalizing it will
    # make it so the NN won't have to figure that out by itself.
    #
    # Strokes is a list of 1+ strokes, and represents one character.
    # A single character can have multiple strokes if you need to lift your pen when writing it,
    # for example i, j, or t.
    #
    # Each stroke is a list of (x,y,t) points, which are represented with dicitonaries.

    # Takes every point from every stroke in strokes and puts them in a single array
    points = np.array(
        [[point["x"], point["y"]] for stroke in strokes for point in stroke], float
    )
    minimum = points.min(0)
    maximum = points.max(0)
    center = (minimum + maximum) / 2
    scale = max((maximum - minimum).max(), 1e-6)

    return [
        np.array(
            [
                [(point["x"] - center[0]) / scale, (point["y"] - center[1]) / scale]
                for point in stroke
            ]
        )
        for stroke in strokes
    ]


def render(
    strokes: NormalizedStrokes, size: int = 32, pad: int = 2, width: int = 2
) -> NDArray[np.float32]:
    img = Image.new("L", (size, size), 0)
    d, s = ImageDraw.Draw(img), size - 2 * pad
    for st in strokes:
        xy = [((x + 0.5) * s + pad, (y + 0.5) * s + pad) for x, y in st]
        if len(xy) > 1:
            d.line(xy, fill=255, width=width)
        else:
            d.point(xy[0], fill=255)
    return np.asarray(img, np.float32) / 255


def count_rows(path: str | Path) -> int:
    with open(path, newline="") as f:
        return sum(1 for _ in f) - 1  # minus the header line


def read_raw_rows(path: str | Path) -> Iterator[RawRow]:
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter=";", quotechar="'"):
            yield int(row["symbol_id"]), row["user_id"], row["data"]


def process_row(row: RawRow) -> Result | None:
    symbol_id, user_id, raw = row
    strokes = [s for s in yaml.load(raw, Loader=LOADER) if s]
    if not strokes:
        return None
    return symbol_id, user_id, render(normalize(strokes))


def build_cache(csv_path: str, out_path: str) -> None:
    print("caching", csv_path)
    images: list[NDArray[np.float32]] = []
    labels: list[int] = []
    users: list[str] = []
    skipped = 0

    with ProcessPoolExecutor() as pool:
        results = pool.map(process_row, read_raw_rows(csv_path), chunksize=500)
        for result in tqdm(results, total=count_rows(csv_path), unit="sample"):
            if result is None:
                skipped += 1
                continue
            symbol_id, user_id, image = result
            images.append(image)
            labels.append(symbol_id)
            users.append(user_id)

    print(f"kept {len(images)}, skipped {skipped}")
    np.savez_compressed(
        out_path,
        images=np.stack(images),
        labels=np.array(labels, dtype=np.int64),
        users=np.array(users, dtype=str),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0

    for images, targets in tqdm(loader, desc="train", leave=False):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        seen += len(targets)

    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        total_loss += F.cross_entropy(logits, targets).item() * len(targets)
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        seen += len(targets)

    return total_loss / seen, correct / seen


def alnum_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, class_ids, symbol_id_map
) -> None:
    # Classes whose LaTeX name is a single letter or digit
    is_alnum = torch.tensor(
        [
            re.fullmatch(r"[A-Za-z0-9]", symbol_id_map[int(sid)]) is not None
            for sid in class_ids
        ]
    )
    print("letter/digit classes:", int(is_alnum.sum()))

    mask = is_alnum[targets]  # samples whose true class is a letter or digit
    print("samples:", int(mask.sum()))

    # 1. Unrestricted: the model may answer with any of the 369 classes
    preds = logits.argmax(dim=1)
    unrestricted = (preds[mask] == targets[mask]).float().mean().item()

    # 2. Restricted: only letters and digits are "enabled"
    restricted_preds = logits.masked_fill(~is_alnum, float("-inf")).argmax(dim=1)
    restricted = (restricted_preds[mask] == targets[mask]).float().mean().item()

    print(f"unrestricted:  {unrestricted:.1%}")
    print(f"letters/digits only: {restricted:.1%}")

    # Weakest classes in the restricted setting
    per_class: list[tuple[float, int]] = []
    for k in torch.nonzero(is_alnum).flatten().tolist():
        selected = targets == k
        if selected.any():
            per_class.append(
                ((restricted_preds[selected] == k).float().mean().item(), k)
            )
    print("weakest:")
    for acc, k in sorted(per_class)[:8]:
        print(f"  {acc:5.0%}  {symbol_id_map[int(class_ids[k])]}")


def show(
    images: NDArray[np.float32],
    symbol_id_map: dict[int, str],
    labels: NDArray[np.int64] | None = None,
) -> None:
    figure, axes = plt.subplots(
        1, len(images), figsize=(2 * len(images), 2), squeeze=False
    )
    for i, (ax, image) in enumerate(zip(axes[0], images)):
        ax.imshow(image, cmap="gray", interpolation="nearest")
        ax.set_title(symbol_id_map[int(labels[i])] if labels is not None else "")
        ax.axis("off")

    figure.savefig("samples.png", dpi=150)
    plt.close(figure)


def summarize(name: str, labels: NDArray[np.int64], users: NDArray[np.str_]) -> None:
    counts = Counter(labels.tolist())
    per_class = np.array(list(counts.values()))
    print(f"--- {name} ---")
    print(f"samples: {len(labels)}")
    print(f"classes: {len(counts)}")
    print(
        f"samples per class: min {per_class.min()}, "
        f"median {int(np.median(per_class))}, max {per_class.max()}"
    )
    print(f"classes with < 10 samples: {int((per_class < 10).sum())}")
    print(f"writers: {len(set(users.tolist()))}")


if __name__ == "__main__":
    main()
