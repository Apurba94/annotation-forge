import json
import os
import random
import shutil
import tempfile
import unittest

from annotation_forge import Box, Dataset, FormatError, Image, clip, leakage, merge, read_coco, read_voc, read_yolo, remap, split, statistics, validate, write_coco, write_voc, write_yolo


def sample(n=10, seed=1):
    rng = random.Random(seed)
    ds = Dataset(classes=["RBC", "WBC", "Platelet"])
    for i in range(n):
        im = Image(f"img{i}.jpg", 640, 480)
        for _ in range(rng.randint(1, 6)):
            w, h = rng.uniform(5, 100), rng.uniform(5, 100)
            im.boxes.append(Box(rng.choice(ds.classes), rng.uniform(0, 640 - w), rng.uniform(0, 480 - h), w, h))
        ds.images.append(im)
    return ds


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def same(self, a, b, places=1):
        self.assertEqual(len(a.images), len(b.images))
        order = {os.path.splitext(i.file)[0]: i for i in b.images}
        for im in a.images:
            other = order[os.path.splitext(im.file)[0]]
            self.assertEqual(len(im.boxes), len(other.boxes))
            for p, q in zip(im.boxes, other.boxes):
                self.assertEqual(p.label, q.label)
                for k in "xywh":
                    self.assertAlmostEqual(getattr(p, k), getattr(q, k), places=places)


class TestFormats(Base):
    def test_coco_round_trip(self):
        ds = sample()
        p = os.path.join(self.dir, "c.json")
        write_coco(ds, p)
        self.same(ds, read_coco(p), 3)

    def test_yolo_round_trip(self):
        ds = sample()
        d = os.path.join(self.dir, "y")
        write_yolo(ds, d)
        back = read_yolo(d, ds.classes, {f"img{i}.jpg": (640, 480) for i in range(10)})
        self.same(ds, back, 2)

    def test_voc_round_trip(self):
        ds = sample()
        d = os.path.join(self.dir, "v")
        write_voc(ds, d)
        self.same(ds, read_voc(d), 3)

    def test_chain_through_every_format(self):
        ds = sample()
        c, y, v = os.path.join(self.dir, "c.json"), os.path.join(self.dir, "y"), os.path.join(self.dir, "v")
        write_coco(ds, c)
        write_yolo(read_coco(c), y)
        a = read_yolo(y, ds.classes, {f"img{i}": (640, 480) for i in range(10)})
        write_voc(a, v)
        self.same(ds, read_voc(v), 1)

    def test_yolo_numbers(self):
        ds = Dataset(classes=["cell"])
        ds.images.append(Image("a.jpg", 200, 100, [Box("cell", 40, 20, 80, 40)]))
        d = os.path.join(self.dir, "y")
        write_yolo(ds, d)
        with open(os.path.join(d, "a.txt")) as f:
            self.assertEqual(f.read().strip(), "0 0.400000 0.400000 0.400000 0.400000")

    def test_yolo_errors(self):
        d = os.path.join(self.dir, "y")
        os.makedirs(d)
        with open(os.path.join(d, "a.txt"), "w") as f:
            f.write("0 0.5 0.5\n")
        for content, sizes in (("0 0.5 0.5\n", {"a": (10, 10)}), ("9 .5 .5 .1 .1\n", {"a": (10, 10)}), ("x .5 .5 .1 .1\n", {"a": (10, 10)}), ("0 .5 .5 .1 .1\n", {})):
            with open(os.path.join(d, "a.txt"), "w") as f:
                f.write(content)
            with self.assertRaises(FormatError):
                read_yolo(d, ["c"], sizes)

    def test_coco_errors(self):
        p = os.path.join(self.dir, "c.json")
        for bad in ({"images": []}, {"images": [], "categories": [], "annotations": [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1]}]}):
            with open(p, "w") as f:
                json.dump(bad, f)
            with self.assertRaises(FormatError):
                read_coco(p)

    def test_broken_xml(self):
        with open(os.path.join(self.dir, "a.xml"), "w") as f:
            f.write("<annotation><size>")
        with self.assertRaises(FormatError):
            read_voc(self.dir)

    def test_bangla_labels_survive(self):
        ds = Dataset(classes=["কোষ"])
        ds.images.append(Image("a.jpg", 10, 10, [Box("কোষ", 1, 1, 2, 2)]))
        p = os.path.join(self.dir, "c.json")
        write_coco(ds, p)
        self.assertEqual(read_coco(p).images[0].boxes[0].label, "কোষ")


class TestValidate(unittest.TestCase):
    def issues(self, *boxes, classes=("a",)):
        ds = Dataset(classes=list(classes))
        ds.images.append(Image("x.jpg", 100, 100, list(boxes)))
        return [(s, m) for s, _, m in validate(ds)]

    def test_clean(self):
        self.assertEqual(self.issues(Box("a", 10, 10, 20, 20)), [])

    def test_problems(self):
        self.assertIn("no area", self.issues(Box("a", 1, 1, 0, 5))[0][1])
        self.assertIn("leaves", self.issues(Box("a", 90, 90, 20, 20))[0][1])
        self.assertIn("unknown class", self.issues(Box("b", 1, 1, 5, 5))[0][1])
        self.assertEqual(self.issues(Box("a", 1, 1, 1, 1))[0][0], "warning")
        self.assertIn("duplicates", self.issues(Box("a", 1, 1, 9, 9), Box("a", 1, 1, 9, 9))[0][1])

    def test_small_rounding_is_tolerated(self):
        self.assertEqual(self.issues(Box("a", -0.2, 0, 100.3, 50)), [])

    def test_repeated_files(self):
        ds = Dataset(classes=["a"])
        ds.images += [Image("x.jpg", 9, 9, [Box("a", 0, 0, 5, 5)]), Image("x.jpg", 9, 9, [Box("a", 0, 0, 5, 5)])]
        self.assertEqual(validate(ds)[0][0], "error")


class TestEdit(unittest.TestCase):
    def test_clip(self):
        ds = Dataset(classes=["a"])
        ds.images.append(Image("x", 100, 100, [Box("a", -10, 90, 30, 30), Box("a", 200, 200, 5, 5)]))
        self.assertEqual(clip(ds), 1)
        b = ds.images[0].boxes[0]
        self.assertEqual((b.x, b.y, b.w, b.h), (0, 90, 20, 10))

    def test_remap(self):
        ds = Dataset(classes=["cat", "kitten", "dog"])
        ds.images.append(Image("x", 9, 9, [Box("cat", 0, 0, 1, 1), Box("kitten", 0, 0, 1, 1), Box("dog", 0, 0, 1, 1)]))
        remap(ds, {"kitten": "cat", "dog": None})
        self.assertEqual(ds.classes, ["cat"])
        self.assertEqual([b.label for b in ds.images[0].boxes], ["cat", "cat"])

    def test_merge_and_stats(self):
        ds = merge(sample(3, 1), sample(4, 2))
        s = statistics(ds)
        self.assertEqual(s["images"], 7)
        self.assertEqual(sum(s["sizes"].values()), s["boxes"])
        self.assertEqual(sum(s["classes"].values()), s["boxes"])

    def test_stats_of_nothing(self):
        self.assertEqual(statistics(Dataset())["images"], 0)


class TestSplit(unittest.TestCase):
    def grouped(self, groups=60, seed=3):
        rng = random.Random(seed)
        ds = Dataset(classes=["common", "rare"])
        for g in range(groups):
            for i in range(rng.randint(1, 8)):
                im = Image(f"g{g}_i{i}.png", 100, 100, group=f"patient{g}")
                for _ in range(rng.randint(1, 5)):
                    im.boxes.append(Box("rare" if rng.random() < 0.08 else "common", 1, 1, 5, 5))
                ds.images.append(im)
        return ds

    def test_groups_never_cross(self):
        s = split(self.grouped())
        self.assertEqual(leakage(s), [])

    def test_every_image_lands_once(self):
        ds = self.grouped()
        s = split(ds)
        names = sorted(im.file for part in s.values() for im in part.images)
        self.assertEqual(names, sorted(im.file for im in ds.images))

    def test_ratios_and_class_balance(self):
        ds = self.grouped(200)
        s = split(ds)
        total = len(ds.images)
        self.assertAlmostEqual(len(s["train"].images) / total, 0.7, delta=0.05)
        self.assertAlmostEqual(len(s["test"].images) / total, 0.15, delta=0.05)
        rare = ds.label_counts()["rare"]
        for name, r in (("train", 0.7), ("val", 0.15), ("test", 0.15)):
            self.assertAlmostEqual(s[name].label_counts()["rare"] / rare, r, delta=0.05)

    def test_repeatable_and_seed_matters(self):
        ds = self.grouped()
        a = [im.file for im in split(ds, seed=1)["test"].images]
        b = [im.file for im in split(ds, seed=1)["test"].images]
        c = [im.file for im in split(ds, seed=2)["test"].images]
        self.assertEqual(a, b)
        self.assertNotEqual(sorted(a), sorted(c))

    def test_bad_ratios(self):
        with self.assertRaises(ValueError):
            split(sample(), {"train": 0.9, "test": 0.3})

    def test_images_without_groups_still_split(self):
        s = split(sample(40))
        self.assertEqual(sum(len(p.images) for p in s.values()), 40)

    def test_leakage_finds_each_kind(self):
        a, b = Dataset(), Dataset()
        a.images.append(Image("same.png", 1, 1, group="p1"))
        b.images.append(Image("same.png", 1, 1, group="p1"))
        a.images.append(Image("x.png", 1, 1))
        b.images.append(Image("y.png", 1, 1))
        found = leakage({"train": a, "test": b}, {"x.png": 0b1111, "y.png": 0b1110})
        self.assertEqual(sorted(p[0] for p in found), ["file", "group", "similar"])


if __name__ == "__main__":
    unittest.main()
