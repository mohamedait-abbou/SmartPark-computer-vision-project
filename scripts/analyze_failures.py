"""Compare YOLO predictions with ground-truth labels, image by image.

This is an analysis tool only. It never trains or changes the model.

A predicted box is a HIT when its IoU (overlap) with a true box is >= 0.50.
A true box with no hit is a MISS. A prediction with no true box is a FALSE
ALARM. We also report the IoU of hits, because low IoU explains a low
mAP50-95 even when mAP50 looks good.
"""
import argparse
import json
from pathlib import Path


def read_boxes(path):
    """Return a list of (cx, cy, w, h, confidence) from a YOLO txt file."""
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        values = [float(x) for x in parts[1:5]]
        conf = float(parts[5]) if len(parts) > 5 else 1.0
        boxes.append((*values, conf))
    return boxes


def to_xyxy(box):
    cx, cy, w, h = box[:4]
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a, b):
    ax1, ay1, ax2, ay2 = to_xyxy(a)
    bx1, by1, bx2, by2 = to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def analyse(truth_dir, pred_dir, iou_threshold=0.50, min_conf=0.0):
    misses, false_alarms, hit_ious = [], [], []
    n_true = n_pred = n_hits = 0
    for truth_path in sorted(Path(truth_dir).glob("*.txt")):
        name = truth_path.stem
        truths = read_boxes(truth_path)
        preds = [p for p in read_boxes(Path(pred_dir) / f"{name}.txt")
                 if p[4] >= min_conf]
        n_true += len(truths)
        n_pred += len(preds)

        used = set()
        for true_box in truths:
            scored = [(iou(true_box, p), i) for i, p in enumerate(preds)
                      if i not in used]
            best_iou, best_i = max(scored, default=(0.0, None))
            if best_i is not None and best_iou >= iou_threshold:
                used.add(best_i)
                n_hits += 1
                hit_ious.append(best_iou)
            else:
                misses.append({"image": name, "true_box": list(true_box[:4])})
        for i, pred in enumerate(preds):
            if i not in used:
                false_alarms.append({
                    "image": name,
                    "pred_box": list(pred[:4]),
                    "confidence": round(pred[4], 3),
                })
    hit_ious.sort()
    return {
        "images": len(list(Path(truth_dir).glob("*.txt"))),
        "true_boxes": n_true,
        "predicted_boxes": n_pred,
        "hits": n_hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "hit_iou_mean": round(sum(hit_ious) / len(hit_ious), 3) if hit_ious else 0,
        "hit_iou_median": hit_ious[len(hit_ious) // 2] if hit_ious else 0,
        "hits_below_iou_075": sum(x < 0.75 for x in hit_ious),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True, help="ground-truth labels dir")
    parser.add_argument("--preds", required=True, help="saved prediction labels dir")
    parser.add_argument("--out", required=True, help="where to write the JSON report")
    parser.add_argument("--min-conf", type=float, default=0.0,
                        help="describe failures only at this confidence; "
                             "this does NOT change the evaluation metrics")
    args = parser.parse_args()

    result = analyse(args.truth, args.preds, min_conf=args.min_conf)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))

    print(f"Images: {result['images']} | true boxes: {result['true_boxes']} "
          f"| predicted boxes: {result['predicted_boxes']}")
    print(f"Hits: {result['hits']} | misses: {len(result['misses'])} "
          f"| false alarms: {len(result['false_alarms'])}")
    print(f"Hit IoU: mean {result['hit_iou_mean']}, "
          f"median {result['hit_iou_median']}, "
          f"below 0.75: {result['hits_below_iou_075']}")
    print("Miss images:", ", ".join(m["image"] for m in result["misses"][:15]))
    print("False-alarm images:",
          ", ".join(f["image"] for f in result["false_alarms"][:15]))
    print("Full report:", args.out)
