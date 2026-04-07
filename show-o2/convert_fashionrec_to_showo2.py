#!/usr/bin/env python3
"""Convert FashionRec basic_recommendation data to Show-o2 Stage 2c format.

Produces:
  - MMU JSON: single-turn pairs (image + first question → target item description)
  - T2I JSONL: partial_outfit item description + image path (deduplicated)
"""

import argparse
import glob
import json
import os

import webdataset as wds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True,
                        help="Path to basic_recommendation/ directory")
    parser.add_argument("--item-images-dir", required=True,
                        help="Path to item_images/ directory")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write output JSON/JSONL files")
    parser.add_argument("--validate-images", action="store_true",
                        help="Check that each referenced item image exists on disk")
    return parser.parse_args()


def process_split(tar_files, item_images_dir, validate_images):
    """Process all tar files for one split.

    Returns (mmu_entries, t2i_entries, stats).
    """
    mmu_entries = []
    t2i_entries = []
    t2i_seen_ids = set()
    stats = {
        "total_samples": 0,
        "mmu_entries": 0,
        "t2i_entries": 0,
        "t2i_deduped": 0,
        "skipped_missing_image": 0,
        "skipped_no_partial_outfit": 0,
        "skipped_no_conversation": 0,
    }

    for tar_path in tar_files:
        print(f"Processing {tar_path} ...")
        ds = (wds.WebDataset(tar_path, shardshuffle=False)
              .decode("pil"))

        for sample in ds:
            if "json" not in sample:
                continue
            stats["total_samples"] += 1
            text = sample["json"]

            # Validate structure
            if not text.get("partial_outfit"):
                stats["skipped_no_partial_outfit"] += 1
                continue
            if not text.get("conversation") or len(text["conversation"]) < 2:
                stats["skipped_no_conversation"] += 1
                continue

            # Get the partial outfit item (first one)
            partial_item = text["partial_outfit"][0]
            item_id = partial_item["item_id"]
            img_filename = f"{item_id}.jpg"
            img_abs_path = os.path.join(item_images_dir, img_filename)

            if validate_images and not os.path.exists(img_abs_path):
                stats["skipped_missing_image"] += 1
                continue

            # Get the first human question
            first_question = text["conversation"][0]["value"]

            # Create MMU entries: one per target item
            target_items = text.get("target_items", [])
            for target in target_items:
                mmu_entry = {
                    "image": img_filename,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{first_question}"},
                        {"from": "gpt", "value": target["description"]},
                    ]
                }
                mmu_entries.append(mmu_entry)
                stats["mmu_entries"] += 1

            # Create T2I entry (deduplicated by item_id)
            if item_id not in t2i_seen_ids:
                t2i_seen_ids.add(item_id)
                t2i_entries.append({
                    "path": img_abs_path,
                    "prompt": partial_item["description"],
                })
                stats["t2i_entries"] += 1
            else:
                stats["t2i_deduped"] += 1

    return mmu_entries, t2i_entries, stats


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_tars = sorted(glob.glob(os.path.join(args.data_root, "train", "*.tar")))
    valid_tars = sorted(glob.glob(os.path.join(args.data_root, "valid", "*.tar")))

    print(f"Found {len(train_tars)} train tars, {len(valid_tars)} valid tars")

    splits = [
        ("train", train_tars, "fashionrec_mmu_train.json", "fashionrec_t2i_train.jsonl"),
        ("eval", valid_tars, "fashionrec_mmu_eval.json", "fashionrec_t2i_eval.jsonl"),
    ]

    for split_name, tar_files, mmu_out, t2i_out in splits:
        if not tar_files:
            print(f"No tar files found for {split_name}, skipping.")
            continue

        print(f"\n=== Processing {split_name} split ===")
        mmu_entries, t2i_entries, stats = process_split(
            tar_files, args.item_images_dir, args.validate_images
        )

        # Write MMU JSON
        mmu_path = os.path.join(args.output_dir, mmu_out)
        with open(mmu_path, "w") as f:
            json.dump(mmu_entries, f)
        print(f"Wrote {len(mmu_entries)} MMU entries to {mmu_path}")

        # Write T2I JSONL
        t2i_path = os.path.join(args.output_dir, t2i_out)
        with open(t2i_path, "w") as f:
            for entry in t2i_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"Wrote {len(t2i_entries)} T2I entries to {t2i_path}")

        # Print stats
        print(f"\n--- {split_name} statistics ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
