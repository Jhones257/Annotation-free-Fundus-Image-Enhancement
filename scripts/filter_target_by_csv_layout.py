import argparse
import csv
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Filter generated images (for example *_fake_TB.png) and reorganize them '
            'using the same folder layout found in a CSV image column.'
        )
    )
    parser.add_argument(
        '--source_dir',
        default='results/gfenet_gray_1ch_200e/test_latest/images/target',
        help='Directory with generated images to filter.',
    )
    parser.add_argument(
        '--csv_path',
        default='itapecuru_all_gray_v3.csv',
        help='CSV file containing the image paths.',
    )
    parser.add_argument(
        '--csv_image_column',
        default='image',
        help='CSV column that stores the image path.',
    )
    parser.add_argument(
        '--output_dir',
        default='results/gfenet_gray_1ch_200e/test_latest/images/target_filtered_like_csv',
        help='Destination root where filtered files will be copied.',
    )
    parser.add_argument(
        '--keep_suffix',
        default='_fake_TB.png',
        help='Only files ending with this suffix will be kept.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite destination files if they already exist.',
    )
    return parser.parse_args()


def normalize_rel_path(path_text: str) -> str:
    return path_text.strip().replace('\\', '/').lstrip('/')


def build_csv_index(csv_path: Path, image_column: str):
    """
    Build a lookup by (relative_folder, image_stem) -> csv_relative_path.
    Example key: ('maria_do socorro...-122632', '0ae9bd99...').
    """
    index = {}
    duplicates = 0

    with csv_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f'CSV has no header: {csv_path}')
        if image_column not in reader.fieldnames:
            raise ValueError(f"CSV column '{image_column}' not found in {csv_path}")

        for row in reader:
            raw_rel = row.get(image_column, '')
            rel = normalize_rel_path(raw_rel)
            if not rel:
                continue

            rel_path = Path(rel)
            key = (rel_path.parent.as_posix(), rel_path.stem)
            if key in index:
                duplicates += 1
                continue
            index[key] = rel_path

    return index, duplicates


def collect_source_candidates(source_dir: Path, keep_suffix: str):
    for file_path in source_dir.rglob('*'):
        if file_path.is_file() and file_path.name.endswith(keep_suffix):
            yield file_path


def copy_filtered_images(source_dir: Path, output_dir: Path, csv_index, keep_suffix: str, overwrite: bool):
    copied = 0
    skipped_existing = 0
    skipped_not_in_csv = 0

    for src in collect_source_candidates(source_dir, keep_suffix):
        rel_to_source = src.relative_to(source_dir)
        folder_key = rel_to_source.parent.as_posix()
        stem = src.name[:-len(keep_suffix)]
        key = (folder_key, stem)

        csv_rel_path = csv_index.get(key)
        if csv_rel_path is None:
            skipped_not_in_csv += 1
            continue

        # Use the exact CSV filename (for example hash.jpg), without _fake_TB in output.
        dst = output_dir / csv_rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not overwrite:
            skipped_existing += 1
            continue

        shutil.copy2(src, dst)
        copied += 1

    return copied, skipped_existing, skipped_not_in_csv


def main():
    args = parse_args()
    root = Path.cwd()

    source_dir = (root / args.source_dir).resolve()
    csv_path = (root / args.csv_path).resolve()
    output_dir = (root / args.output_dir).resolve()

    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f'Source directory not found: {source_dir}')
    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f'CSV file not found: {csv_path}')

    csv_index, duplicates = build_csv_index(csv_path, args.csv_image_column)
    copied, skipped_existing, skipped_not_in_csv = copy_filtered_images(
        source_dir=source_dir,
        output_dir=output_dir,
        csv_index=csv_index,
        keep_suffix=args.keep_suffix,
        overwrite=args.overwrite,
    )

    print('Done.')
    print(f'CSV indexed images: {len(csv_index)}')
    print(f'Duplicate image keys ignored in CSV: {duplicates}')
    print(f'Copied files: {copied}')
    print(f'Skipped (already existed): {skipped_existing}')
    print(f'Skipped (suffix matched but not found in CSV): {skipped_not_in_csv}')
    print(f'Output directory: {output_dir}')


if __name__ == '__main__':
    main()
