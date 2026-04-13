import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.get_mask import get_mask

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Copy a fundus dataset into the GFENet dataroot layout and auto-generate masks.'
    )
    parser.add_argument('--input_dir', required=True, help='Root directory that contains the original images.')
    parser.add_argument('--output_dir', required=True,
                        help='Destination directory that will contain target/ and target_mask/ trees.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing copied images and masks.')
    parser.add_argument('--extensions', nargs='+', default=sorted(VALID_EXTENSIONS),
                        help='List of valid image extensions (case-insensitive).')
    parser.add_argument('--csv', default='',
                        help='Optional CSV file with image metadata. If provided, only listed images are prepared.')
    parser.add_argument('--csv_image_column', default='image',
                        help='CSV column containing image paths relative to input_dir.')
    parser.add_argument('--csv_quality_column', default='quality',
                        help='CSV column containing quality labels.')
    parser.add_argument('--qualities', nargs='+', type=int, default=[1, 2],
                        help='Quality values to keep when --csv is provided (default: 1 2).')
    return parser.parse_args()


def is_valid_image(path, allowed_exts):
    return path.suffix.lower() in allowed_exts


def ensure_dataset_scaffold(root: Path):
    for sub in ['source', 'source_mask', 'target', 'target_mask']:
        (root / sub).mkdir(parents=True, exist_ok=True)


def save_mask(mask_array: np.ndarray, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask_array.astype(np.uint8)) * 255).save(destination)


def copy_image(src: Path, dst: Path, overwrite: bool):
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_with_mask(image_path: Path, input_dir: Path, target_dir: Path, target_mask_dir: Path, overwrite: bool):
    relative_path = image_path.relative_to(input_dir)
    destination_image = target_dir / relative_path
    destination_mask = target_mask_dir / relative_path.with_suffix('.png')

    if destination_image.exists() and destination_mask.exists() and not overwrite:
        return False

    copy_image(image_path, destination_image, overwrite)
    pil_image = Image.open(image_path).convert('RGB')
    mask_array = get_mask(pil_image)
    save_mask(mask_array, destination_mask)
    return True


def iter_images_from_csv(input_dir: Path, csv_path: Path, image_column: str, quality_column: str, qualities):
    quality_set = {str(q) for q in qualities}
    with csv_path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f'CSV has no header: {csv_path}')
        if image_column not in reader.fieldnames:
            raise ValueError(f"CSV column '{image_column}' not found in {csv_path}")
        if quality_column not in reader.fieldnames:
            raise ValueError(f"CSV column '{quality_column}' not found in {csv_path}")

        seen = set()
        for row in reader:
            if row.get(quality_column) not in quality_set:
                continue
            rel_path_str = (row.get(image_column) or '').strip().replace('\\', '/')
            if not rel_path_str:
                continue
            if rel_path_str in seen:
                continue
            seen.add(rel_path_str)
            yield input_dir / rel_path_str


def build_dataset(input_dir: Path, output_dir: Path, overwrite: bool, allowed_exts,
                  csv_path: Path = None, image_column: str = 'image', quality_column: str = 'quality', qualities=None):
    ensure_dataset_scaffold(output_dir)
    target_dir = output_dir / 'target'
    target_mask_dir = output_dir / 'target_mask'

    copied = 0
    skipped = 0
    missing = 0
    invalid_ext = 0

    if csv_path is None:
        candidate_images = sorted(input_dir.rglob('*'))
    else:
        candidate_images = sorted(
            iter_images_from_csv(input_dir, csv_path, image_column, quality_column, qualities),
            key=lambda p: str(p)
        )

    for image_path in candidate_images:
        if not image_path.is_file():
            missing += 1
            continue
        if not is_valid_image(image_path, allowed_exts):
            invalid_ext += 1
            continue

        copied_now = _copy_with_mask(image_path, input_dir, target_dir, target_mask_dir, overwrite)
        if copied_now:
            copied += 1
        else:
            skipped += 1

    print(f'Finished! Copied {copied} images. Skipped {skipped} already existing files.')
    if csv_path is not None:
        print(f'CSV filtering active: qualities={qualities}')
        print(f'Missing files from CSV: {missing}')
        print(f'Invalid extension (ignored): {invalid_ext}')


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    allowed_exts = {ext.lower() for ext in args.extensions}
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else None

    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory not found: {input_dir}')
    if csv_path is not None and not csv_path.exists():
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    build_dataset(
        input_dir,
        output_dir,
        args.overwrite,
        allowed_exts,
        csv_path=csv_path,
        image_column=args.csv_image_column,
        quality_column=args.csv_quality_column,
        qualities=args.qualities,
    )


if __name__ == '__main__':
    main()
