"""One in-memory model for detection labels, with readers and writers for each format.

Coordinates inside are pixels: x, y of the top-left corner, then width and height.
"""
import json
import os
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field


class FormatError(Exception):
    pass


@dataclass
class Box:
    label: str
    x: float
    y: float
    w: float
    h: float
    difficult: bool = False


@dataclass
class Image:
    file: str
    width: int
    height: int
    boxes: list = field(default_factory=list)
    group: str = None           # patient, slide or video the image came from


@dataclass
class Dataset:
    images: list = field(default_factory=list)
    classes: list = field(default_factory=list)

    def label_counts(self):
        return Counter(b.label for im in self.images for b in im.boxes)

    def ensure_classes(self):
        for name in sorted(self.label_counts()):
            if name not in self.classes:
                self.classes.append(name)


def read_yolo(folder, classes, sizes):
    """folder holds one .txt per image. sizes: {stem or file name: (width, height)}."""
    ds = Dataset(classes=list(classes))
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".txt") or name == "classes.txt":
            continue
        stem = name[:-4]
        size = sizes.get(stem) or next((v for k, v in sizes.items() if os.path.splitext(k)[0] == stem), None)
        if size is None:
            raise FormatError(f"no image size for {stem}")
        w, h = size
        im = Image(stem, w, h)
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            for number, line in enumerate(f, 1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    raise FormatError(f"{name} line {number}: expected 5 values")
                try:
                    c, cx, cy, bw, bh = int(parts[0]), *(float(v) for v in parts[1:])
                except ValueError:
                    raise FormatError(f"{name} line {number}: not a number") from None
                if not 0 <= c < len(classes):
                    raise FormatError(f"{name} line {number}: class {c} is not in the class list")
                im.boxes.append(Box(classes[c], (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h))
        ds.images.append(im)
    return ds


def write_yolo(ds, folder):
    os.makedirs(folder, exist_ok=True)
    ds.ensure_classes()
    index = {c: i for i, c in enumerate(ds.classes)}
    for im in ds.images:
        stem = os.path.splitext(os.path.basename(im.file))[0]
        with open(os.path.join(folder, stem + ".txt"), "w", encoding="utf-8", newline="\n") as f:
            for b in im.boxes:
                f.write(f"{index[b.label]} {(b.x + b.w / 2) / im.width:.6f} {(b.y + b.h / 2) / im.height:.6f} {b.w / im.width:.6f} {b.h / im.height:.6f}\n")
    with open(os.path.join(folder, "classes.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(ds.classes) + "\n")


def read_coco(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise FormatError(f"COCO file has no {key}")
    names = {c["id"]: c["name"] for c in data["categories"]}
    ds = Dataset(classes=[c["name"] for c in sorted(data["categories"], key=lambda c: c["id"])])
    by_id = {}
    for im in data["images"]:
        by_id[im["id"]] = Image(im["file_name"], im["width"], im["height"])
        ds.images.append(by_id[im["id"]])
    for a in data["annotations"]:
        if a["image_id"] not in by_id:
            raise FormatError(f"annotation {a.get('id')} points to a missing image")
        if a["category_id"] not in names:
            raise FormatError(f"annotation {a.get('id')} has an unknown category")
        x, y, w, h = a["bbox"]
        by_id[a["image_id"]].boxes.append(Box(names[a["category_id"]], x, y, w, h))
    return ds


def write_coco(ds, path):
    ds.ensure_classes()
    index = {c: i + 1 for i, c in enumerate(ds.classes)}
    images, anns = [], []
    for i, im in enumerate(ds.images, 1):
        images.append({"id": i, "file_name": im.file, "width": im.width, "height": im.height})
        for b in im.boxes:
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": index[b.label], "bbox": [round(v, 4) for v in (b.x, b.y, b.w, b.h)], "area": round(b.w * b.h, 4), "iscrowd": 0})
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": anns, "categories": [{"id": i, "name": c} for c, i in index.items()]}, f, indent=1, ensure_ascii=False)


def read_voc(folder):
    ds = Dataset()
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".xml"):
            continue
        try:
            root = ET.parse(os.path.join(folder, name)).getroot()
        except ET.ParseError as exc:
            raise FormatError(f"{name}: {exc}") from None
        size = root.find("size")
        if size is None:
            raise FormatError(f"{name}: no size element")
        im = Image(root.findtext("filename") or name[:-4], int(float(size.findtext("width"))), int(float(size.findtext("height"))))
        for obj in root.findall("object"):
            b = obj.find("bndbox")
            x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            label = obj.findtext("name")
            im.boxes.append(Box(label, x1, y1, x2 - x1, y2 - y1, obj.findtext("difficult") == "1"))
            if label not in ds.classes:
                ds.classes.append(label)
        ds.images.append(im)
    ds.classes.sort()
    return ds


def write_voc(ds, folder):
    os.makedirs(folder, exist_ok=True)
    for im in ds.images:
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = im.file
        size = ET.SubElement(root, "size")
        for k, v in (("width", im.width), ("height", im.height), ("depth", 3)):
            ET.SubElement(size, k).text = str(v)
        for b in im.boxes:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = b.label
            ET.SubElement(obj, "difficult").text = "1" if b.difficult else "0"
            box = ET.SubElement(obj, "bndbox")
            for k, v in (("xmin", b.x), ("ymin", b.y), ("xmax", b.x + b.w), ("ymax", b.y + b.h)):
                ET.SubElement(box, k).text = f"{v:.4f}"
        ET.ElementTree(root).write(os.path.join(folder, os.path.splitext(os.path.basename(im.file))[0] + ".xml"), encoding="utf-8")


def validate(ds, min_size=2, duplicate_iou=0.95):
    """Return a list of (severity, image, message). Errors break training, warnings deserve a look."""
    issues = []
    seen = Counter(im.file for im in ds.images)
    for file, n in seen.items():
        if n > 1:
            issues.append(("error", file, f"appears {n} times"))
    for im in ds.images:
        if im.width <= 0 or im.height <= 0:
            issues.append(("error", im.file, "image has no size"))
            continue
        if not im.boxes:
            issues.append(("info", im.file, "no boxes"))
        for i, b in enumerate(im.boxes):
            if b.w <= 0 or b.h <= 0:
                issues.append(("error", im.file, f"box {i} has no area"))
                continue
            if b.x < -0.5 or b.y < -0.5 or b.x + b.w > im.width + 0.5 or b.y + b.h > im.height + 0.5:
                issues.append(("error", im.file, f"box {i} leaves the image"))
            if b.w < min_size or b.h < min_size:
                issues.append(("warning", im.file, f"box {i} is only {b.w:.1f}×{b.h:.1f} pixels"))
            if b.label not in ds.classes:
                issues.append(("error", im.file, f"box {i} has unknown class {b.label}"))
            for j in range(i):
                o = im.boxes[j]
                if o.w > 0 and o.h > 0 and o.label == b.label and _iou(o, b) >= duplicate_iou:
                    issues.append(("warning", im.file, f"boxes {j} and {i} are duplicates"))
    return issues


def _iou(a, b):
    w = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    h = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    i = w * h
    u = a.w * a.h + b.w * b.h - i
    return i / u if u > 0 else 0


def clip(ds):
    """Trim boxes to the image and drop those left with no area. Returns how many were dropped."""
    dropped = 0
    for im in ds.images:
        kept = []
        for b in im.boxes:
            x1, y1 = max(b.x, 0), max(b.y, 0)
            x2, y2 = min(b.x + b.w, im.width), min(b.y + b.h, im.height)
            if x2 > x1 and y2 > y1:
                kept.append(Box(b.label, x1, y1, x2 - x1, y2 - y1, b.difficult))
            else:
                dropped += 1
        im.boxes = kept
    return dropped


def remap(ds, mapping):
    """Rename or merge classes. A class mapped to None is removed."""
    for im in ds.images:
        kept = []
        for b in im.boxes:
            new = mapping.get(b.label, b.label)
            if new is not None:
                b.label = new
                kept.append(b)
        im.boxes = kept
    out = []
    for c in ds.classes:
        new = mapping.get(c, c)
        if new is not None and new not in out:
            out.append(new)
    ds.classes = out


def merge(*sets):
    out = Dataset()
    for s in sets:
        for c in s.classes:
            if c not in out.classes:
                out.classes.append(c)
        out.images.extend(s.images)
    return out


def statistics(ds):
    areas = [b.w * b.h for im in ds.images for b in im.boxes]
    counts = ds.label_counts()
    sizes = {"small": sum(a < 32 ** 2 for a in areas), "medium": sum(32 ** 2 <= a < 96 ** 2 for a in areas), "large": sum(a >= 96 ** 2 for a in areas)}
    return {"images": len(ds.images), "boxes": len(areas), "empty_images": sum(1 for im in ds.images if not im.boxes), "boxes_per_image": len(areas) / len(ds.images) if ds.images else 0, "classes": dict(counts), "sizes": sizes, "imbalance": max(counts.values()) / min(counts.values()) if counts else 0}


def split(ds, ratios=None, seed=0):
    """Stratified group split. Images from one group always stay together, which is what stops
    leakage, and groups are assigned greedily to whichever split most needs their classes.
    ratios: {"train": 0.7, "val": 0.15, "test": 0.15}. Returns {name: Dataset}."""
    ratios = ratios or {"train": 0.7, "val": 0.15, "test": 0.15}
    total = sum(ratios.values())
    if abs(total - 1) > 1e-9:
        raise ValueError("ratios must sum to 1")
    groups = defaultdict(list)
    for i, im in enumerate(ds.images):
        groups[im.group or f"__image_{i}"].append(im)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    profile = {k: Counter(b.label for im in groups[k] for b in im.boxes) for k in keys}
    want_label = ds.label_counts()
    want_images = len(ds.images)
    have_label = {n: Counter() for n in ratios}
    have_images = dict.fromkeys(ratios, 0)
    keys.sort(key=lambda k: -sum(profile[k].values()))     # the largest groups are hardest to place

    def need(name, k):
        """How much putting this group here helps: positive when the split lacks these classes."""
        r = ratios[name]
        score = 0.0
        for label, n in profile[k].items():
            score += (r * want_label[label] - have_label[name][label]) / max(want_label[label], 1) * n
        return score + (r * want_images - have_images[name]) / max(want_images, 1)
    out = {n: Dataset(classes=list(ds.classes)) for n in ratios}
    for k in keys:
        best = max(ratios, key=lambda name: (need(name, k), ratios[name]))
        have_label[best].update(profile[k])
        have_images[best] += len(groups[k])
        out[best].images.extend(groups[k])
    return out


def leakage(splits, hashes=None, near=4):
    """Find what crosses split boundaries: shared files, groups, or look-alike images."""
    problems = []
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            fa, fb = {im.file for im in splits[a].images}, {im.file for im in splits[b].images}
            for f in sorted(fa & fb):
                problems.append(("file", a, b, f))
            ga = {im.group for im in splits[a].images if im.group}
            gb = {im.group for im in splits[b].images if im.group}
            for g in sorted(ga & gb):
                problems.append(("group", a, b, g))
            if hashes:
                for x in fa:
                    for y in fb:
                        if x in hashes and y in hashes and (hashes[x] ^ hashes[y]).bit_count() <= near:
                            problems.append(("similar", a, b, f"{x} ~ {y}"))
    return problems
