"""Check or convert labels.

python -m annotation_forge check coco annotations.json
python -m annotation_forge check voc xml_folder
python -m annotation_forge convert coco annotations.json yolo out_folder
python -m annotation_forge convert voc xml_folder coco out.json
"""
import json
import sys

from . import read_coco, read_voc, statistics, validate, write_coco, write_voc, write_yolo

READ = {"coco": read_coco, "voc": read_voc}
WRITE = {"coco": write_coco, "voc": write_voc, "yolo": write_yolo}


def main(argv):
    if len(argv) >= 4 and argv[1] == "check":
        ds = READ[argv[2]](argv[3])
        print(json.dumps(statistics(ds), indent=1, ensure_ascii=False))
        issues = validate(ds)
        for severity, file, text in issues[:200]:
            print(f"{severity:<8}{file}: {text}")
        return 1 if any(i[0] == "error" for i in issues) else 0
    if len(argv) >= 6 and argv[1] == "convert":
        ds = READ[argv[2]](argv[3])
        WRITE[argv[4]](ds, argv[5])
        print(f"wrote {len(ds.images)} images and {sum(len(i.boxes) for i in ds.images)} boxes")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
