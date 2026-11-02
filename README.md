# Annotation Forge

Convert detection labels between YOLO, COCO and Pascal VOC, check them for mistakes, and split them without leaking data. It uses only the Python standard library.

```
python -m annotation_forge check coco annotations.json
python -m annotation_forge convert voc xml_folder coco annotations.json
python -m annotation_forge convert coco annotations.json yolo labels
```

```python
from annotation_forge import leakage, read_coco, split, validate, write_yolo

ds = read_coco("cells.json")
for image in ds.images:
    image.group = image.file.split("_")[0]        # for example the slide or patient id

parts = split(ds, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
assert leakage(parts) == []
write_yolo(parts["train"], "labels/train")
```

## Why group splitting matters

Images from one patient, slide or video look alike. If they land in both train and test, scores look better than they are. `split` keeps each group whole and still balances classes across the splits, including rare classes, by placing each group in the split that most needs it. `leakage` then checks for shared files, shared groups and, if you give it perceptual hashes, look-alike images.

## What the checks find

| Severity | Examples |
|---|---|
| Error | Boxes with no area, boxes leaving the image, unknown classes, repeated files, images with no size |
| Warning | Tiny boxes, duplicate boxes |
| Info | Images with no boxes |

There are also tools to clip boxes to the image, rename or merge classes (`remap`), combine datasets and summarise class balance and object sizes.

## How it is tested

Every format is written and read back, and a chain through COCO, YOLO and VOC must preserve boxes. Bad input must raise a clear error. Splitting is checked for no leakage, every image placed once, target ratios within 5 points, and a rare class spread in proportion.

## Limits

It handles bounding boxes only, not masks or keypoints. YOLO files store no image size, so you must supply sizes when reading them. The YOLO format rounds to 6 decimals, so a round trip can move a box by a fraction of a pixel. Pairwise leakage search with hashes is slow on very large datasets.

## Tests

```
python -m unittest discover -s tests
```

MIT licence.
