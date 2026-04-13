#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare training data for GFE-Net fine-tuning using the Itapecuru dataset.

This script:
  1. Reads the CSV with quality/tipo annotations
  2. Filters quality=0 (good) + tipo=color images as clean references
  3. Applies synthetic degradations to create paired training data
  4. Generates side-by-side [degraded | clean] images for source/
  5. Creates binary masks for source_mask/
  6. Copies quality=2 (rejected) images to target/ for validation

The degradation algorithms are adapted from data/get_low_quality/ but
parameterized for arbitrary resolution (default 768x768).

Usage:
    python scripts/prepare_gfenet_training.py \
        --csv itapecuru_all_labeled_v3_com_tipo.csv \
        --images_dir ORIGINAL_CLEAN_768x768 \
        --output_dir datasets/gfenet_finetune \
        --num_degradations 8 \
        --image_size 768
"""

import argparse
import csv
import math
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from scipy import ndimage

# ---------------------------------------------------------------------------
# Project root so we can import util.get_mask
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.get_mask import get_mask  # returns boolean ndarray

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 2024
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Degradation types (same coding as original repo)
#   Bit positions: [BLUR, SPOT, ILLUMINATION, CATARACT]
#   '1000' = blur only, '0100' = spots only, '0010' = illumination only,
#   '0110' = spots+illum, '1010' = blur+illum, '1100' = blur+spots,
#   '1110' = blur+spots+illum, '0001' = cataract simulation
# ---------------------------------------------------------------------------
DEGRADATION_TYPES = [
    '1000',  # blur
    '0100',  # spots
    '0010',  # illumination
    '0110',  # spots + illumination
    '1010',  # blur + illumination
    '1100',  # blur + spots
    '1110',  # blur + spots + illumination
    '0001',  # cataract
]


# ===== Degradation Functions (adapted for arbitrary resolution) ============

def pil_to_chw(img_pil, size):
    """Convert PIL Image to numpy (C, H, W) float [0, 1]."""
    img = img_pil.resize((size, size), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, C)
    return arr.transpose(2, 0, 1)  # (C, H, W)


def chw_to_uint8(arr):
    """Convert (C, H, W) float [0, 1] → (H, W, C) uint8 [0, 255]."""
    arr = np.clip(arr, 0, 1)
    return (arr.transpose(1, 2, 0) * 255).astype(np.uint8)


def de_color(img_pil, brightness=0.3, contrast=0.4, saturation=0.4):
    """Randomly adjust brightness, contrast, saturation. Returns (C,H,W) float."""
    bf = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness - 0.1)
    cf = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
    sf = random.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation)

    img = ImageEnhance.Brightness(img_pil).enhance(bf)
    img = ImageEnhance.Contrast(img).enhance(cf)
    img = ImageEnhance.Color(img).enhance(sf)

    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1), bf


def de_halo(img, h, w, brightness_factor):
    """Add ring-like halo artifact. img: (C, H, W) float [0, 1]."""
    weight_r = [251 / 255, 141 / 255, 177 / 255]
    weight_g = [249 / 255, 238 / 255, 195 / 255]
    weight_b = [246 / 255, 238 / 255, 147 / 255]

    num = random.randint(1, 2) if brightness_factor >= 0.2 else random.randint(0, 2)

    w0_a = random.randint(w // 2 - w // 8, w // 2 + w // 8)
    h0_a = random.randint(h // 2 - h // 8, h // 2 + h // 8)
    center_a = [w0_a, h0_a]

    wei_dia_a = 0.75 + 0.25 * random.random()
    dia_a = min(h, w) * wei_dia_a
    Y_a, X_a = np.ogrid[:h, :w]
    dist_a = np.sqrt((X_a - center_a[0]) ** 2 + (Y_a - center_a[1]) ** 2)

    mask_a = np.zeros((h, w))
    mask_a[dist_a <= dia_a / 2] = np.mean(img)

    center_b = center_a
    Y_b, X_b = np.ogrid[:h, :w]
    dist_b = np.sqrt((X_b - center_b[0]) ** 2 + (Y_b - center_b[1]) ** 2)

    dia_b_max = 2 * int(np.sqrt(
        max(center_a[0], h - center_a[0]) ** 2 +
        max(center_a[1], w - center_a[1]) ** 2
    )) / min(w, h)
    wei_dia_b = 1.0 + (dia_b_max - 1.0) * random.random()

    if num == 0:
        dia_b = min(h, w) * wei_dia_b + abs(
            max(center_b[0] - w / 2, center_b[1] - h / 2) + max(w, h) * 2 / 3)
    else:
        dia_b = min(h, w) * wei_dia_b + abs(
            max(center_b[0] - w / 2, center_b[1] - h / 2) + max(w, h) / 2)

    mask_b = np.zeros((h, w))
    mask_b[dist_b <= dia_b / 2] = np.mean(img)

    delta_circle = np.abs(mask_a - mask_b) * 1.0
    dia = max(center_a[0], h - center_a[0], center_a[1], w - center_a[1]) * 2
    gauss_rad = max(3, int(np.abs(dia - dia_a)))
    sigma = 2 / 3 * gauss_rad
    if gauss_rad % 2 == 0:
        gauss_rad += 1

    delta_circle = cv2.GaussianBlur(delta_circle, (gauss_rad, gauss_rad), sigma)
    delta_circle = np.array([
        weight_r[num] * delta_circle,
        weight_g[num] * delta_circle,
        weight_b[num] * delta_circle,
    ])

    img = np.clip(img + delta_circle, 0, 1)
    return img


def de_hole(img, h, w, region_mask, d_min=0.4, d_max=0.7):
    """Add dark-spot illumination artifact. img: (C, H, W) float [0, 1]."""
    diameter = random.randint(int(d_min * w), int(d_max * w))
    center = [random.randint(w // 4, w * 3 // 4), random.randint(h * 3 // 8, h * 5 // 8)]

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    circle = dist <= (diameter / 2)

    mask = np.zeros((h, w))
    mask[circle] = 1

    num_valid = max(np.sum(region_mask), 1)
    aver_color = np.sum(img) / (3 * num_valid)
    if aver_color > 0.25:
        brightness = random.uniform(-0.262, -0.26)
        bf = random.uniform(brightness - 0.06 * aver_color, brightness - 0.05 * aver_color)
    else:
        bf = 0

    mask = mask * bf
    rad_w = random.randint(int(diameter * 0.55), int(diameter * 0.75))
    rad_h = random.randint(int(diameter * 0.55), int(diameter * 0.75))
    sigma = 2 / 3 * max(rad_h, rad_w) * 1.2
    if rad_w % 2 == 0:
        rad_w += 1
    if rad_h % 2 == 0:
        rad_h += 1

    mask = cv2.GaussianBlur(mask, (rad_w, rad_h), sigma)
    mask = np.array([mask, mask, mask])
    img = np.clip(img + mask, 0, 1)
    return img


def de_illumination(img_pil, region_mask, h, w):
    """Combined color+halo+hole illumination degradation."""
    img, bf = de_color(img_pil, brightness=0.5, contrast=0.5, saturation=0.5)
    img = de_halo(img, h, w, bf)
    img = de_hole(img, h, w, region_mask, d_min=0.5, d_max=0.8)
    return img


def de_spot(img, h, w, s_max_num=10):
    """Add random spot artifacts. img: (C, H, W) float [0, 1]."""
    s_num = random.randint(5, s_max_num)
    for _ in range(s_num):
        radius = random.randint(max(1, math.ceil(0.01 * h)), max(2, int(0.05 * h)))
        center = [
            random.randint(radius + 1, w - radius - 1),
            random.randint(radius + 1, h - radius - 1),
        ]
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        circle = dist <= (radius / 2)

        k = (14 / 25) + (1.0 - radius / 25)
        beta = 0.5 + (1.5 - 0.5) * (radius / 25)
        A = k * np.ones((3, 1))
        d = 0.3 * (radius / 25)
        t = math.exp(-beta * d)

        mask = np.zeros((h, w))
        mask[circle] = A[0, 0] * (1 - t)

        sigma = (5 + 20 * (radius / 25)) * 2
        rad_w = max(3, random.randint(max(1, int(sigma / 5)), max(2, int(sigma / 4))))
        rad_h = max(3, random.randint(max(1, int(sigma / 5)), max(2, int(sigma / 4))))
        if rad_w % 2 == 0:
            rad_w += 1
        if rad_h % 2 == 0:
            rad_h += 1

        mask = cv2.GaussianBlur(mask, (rad_w, rad_h), sigma)
        mask = np.array([mask, mask, mask])
        img = np.clip(img + mask, 0, 1)

    return img


def de_blur(img, h, w, ratio=3):
    """Gaussian blur degradation. img: (C, H, W) float [0, 1]."""
    img_hwc = img.transpose(1, 2, 0)
    sigma = 5 + (15 - 5) * random.random() * ratio
    rad_w = max(3, random.randint(max(1, int(sigma / 3)), max(2, int(sigma / 2))))
    rad_h = max(3, random.randint(max(1, int(sigma / 3)), max(2, int(sigma / 2))))
    if rad_w % 2 == 0:
        rad_w += 1
    if rad_h % 2 == 0:
        rad_h += 1

    img_hwc = cv2.GaussianBlur(img_hwc, (rad_w, rad_h), sigma)
    return np.clip(img_hwc.transpose(2, 0, 1), 0, 1)


def gaussian_5x5(img):
    """5x5 Gaussian kernel convolution for cataract simulation."""
    kernel = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1],
    ], dtype=np.float64)
    kernel = kernel / kernel.sum()
    return ndimage.convolve(img, kernel)


def cataract_simulation(img_bgr, mask_2d, size):
    """
    Simulate cataract-like haze on a fundus image.

    Args:
        img_bgr: BGR image (H, W, 3) uint8
        mask_2d: binary mask (H, W) float [0, 1]
        size:    target (w, h) tuple

    Returns:
        degraded (H, W, 3) uint8, clean (H, W, 3) uint8
    """
    im_A = cv2.resize(img_bgr, size)
    h, w, c = im_A.shape
    mask_A = mask_2d
    mask_A_3 = (mask_A / max(mask_A.max(), 1e-6))[:, :, np.newaxis]

    # Random center offset for haze
    wp = random.randint(int(-w * 0.3), int(w * 0.3))
    hp = random.randint(int(-h * 0.3), int(h * 0.3))
    transmap = np.ones((h, w))
    cy, cx = h // 2 + hp, w // 2 + wp
    cy = max(0, min(h - 1, cy))
    cx = max(0, min(w - 1, cx))
    transmap[cy, cx] = 0
    transmap = gaussian_5x5(ndimage.distance_transform_edt(transmap)) * mask_A
    if transmap.max() > 0:
        transmap = transmap / transmap.max()

    sum_map = transmap
    if sum_map.max() > 0:
        sum_map = sum_map / sum_map.max()

    # Random blur
    randomR = random.choice([1, 3, 5, 7])
    randomS = random.randint(10, 30)
    fundus_blur = cv2.GaussianBlur(im_A, (randomR, randomR), randomS)

    B, G, R = cv2.split(fundus_blur)
    panel = cv2.merge([
        sum_map * (B.max() - B),
        sum_map * (G.max() - G),
        sum_map * (R.max() - R),
    ])

    panel_ratio = random.uniform(0.6, 0.8)
    sum_degrad = 0.8 * fundus_blur + panel * panel_ratio
    sum_degrad = np.clip(sum_degrad, 0, 255)

    # Color augmentation
    c_ratio = random.uniform(0.9, 1.3)
    b_ratio = random.uniform(0.9, 1.0)
    e_ratio = random.uniform(0.9, 1.3)
    img = Image.fromarray(sum_degrad.astype('uint8'))
    img = ImageEnhance.Contrast(img).enhance(c_ratio)
    img = ImageEnhance.Brightness(img).enhance(b_ratio)
    img = ImageEnhance.Color(img).enhance(e_ratio)

    degraded = (np.array(img, dtype=np.float64) * mask_A_3).astype(np.uint8)
    return degraded, im_A


def apply_degradation(img_pil, mask_bool, size, deg_type):
    """
    Apply a degradation type to a clean image.

    Args:
        img_pil:   PIL Image RGB
        mask_bool: boolean mask (H, W)
        size:      int, target resolution (e.g., 768)
        deg_type:  str, 4-char code like '1010'

    Returns:
        degraded_uint8: (H, W, 3) uint8 BGR
        clean_uint8:    (H, W, 3) uint8 BGR
    """
    h, w = size, size
    img_resized = img_pil.resize((w, h), Image.BICUBIC)

    # Masks in different formats
    mask_float_2d = cv2.resize(
        mask_bool.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
    )
    mask_chw = mask_float_2d[np.newaxis, :, :]  # (1, H, W)

    # Clean image in BGR for cataract simulation
    clean_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

    if deg_type == '0001':
        # Cataract simulation
        degraded_bgr, clean_bgr_out = cataract_simulation(
            clean_bgr, mask_float_2d, (w, h)
        )
        return degraded_bgr, clean_bgr_out

    # For non-cataract types, parse the 3-char degradation code
    code = deg_type[:3]  # blur-spot-illumination

    if code[2] == '1':
        # Illumination includes color transform: starts from PIL
        img_chw = de_illumination(img_resized, mask_chw, h, w)
    else:
        img_chw = pil_to_chw(img_resized, size)

    if code[1] == '1':
        if code[2] == '0':
            # If illumination not applied, need to convert to CHW first
            img_chw = pil_to_chw(img_resized, size) if code[2] == '0' else img_chw
        img_chw = de_spot(img_chw, h, w, s_max_num=10)

    if code[0] == '1':
        if code[1] == '0' and code[2] == '0':
            img_chw = pil_to_chw(img_resized, size)
        img_chw = de_blur(img_chw, h, w, ratio=3)

    # Apply mask and convert
    deg_hwc = (img_chw * mask_chw).transpose(1, 2, 0)
    degraded_uint8 = np.clip(deg_hwc * 255, 0, 255).astype(np.uint8)
    degraded_bgr = cv2.cvtColor(degraded_uint8, cv2.COLOR_RGB2BGR)

    return degraded_bgr, clean_bgr


# ===== Dataset Scaffold / IO ================================================

def ensure_scaffold(output_dir):
    """Create the GFENet dataroot directory structure."""
    for sub in ['source', 'source_mask', 'target', 'target_mask']:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)


def save_mask_png(mask_bool, path):
    """Save boolean mask as 0/255 PNG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((mask_bool.astype(np.uint8) * 255)).save(path)


def make_flat_name(patient_folder, image_name, suffix=''):
    """
    Create a flat filename from patient folder + image name.
    The dataset loader uses os.path.split(path)[-1] to find the mask,
    so filenames must be unique and flat (no subdirectories).
    """
    # Remove extension from image_name
    base = os.path.splitext(image_name)[0]
    # Clean patient folder (replace spaces and special chars)
    patient_clean = patient_folder.replace(' ', '_').replace('/', '_').replace('\\', '_')
    return f"{patient_clean}__{base}{suffix}"


# ===== Main =================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare GFE-Net fine-tuning training data from Itapecuru dataset.'
    )
    parser.add_argument('--csv', required=True,
                        help='Path to itapecuru_all_labeled_v3_com_tipo.csv')
    parser.add_argument('--images_dir', required=True,
                        help='Path to ORIGINAL_CLEAN_768x768/')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory (GFENet dataroot)')
    parser.add_argument('--num_degradations', type=int, default=8,
                        help='Number of degradation variants per image (max 8)')
    parser.add_argument('--image_size', type=int, default=768,
                        help='Target image resolution')
    parser.add_argument('--quality_train', nargs='+', type=int, default=[0],
                        help='Quality values for clean training images (default: 0=good)')
    parser.add_argument('--quality_target', nargs='+', type=int, default=[2],
                        help='Quality values for target/validation images (default: 2=rejected)')
    parser.add_argument('--tipo', nargs='+', default=['color'],
                        help='Image types to include (default: color)')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output files')
    return parser.parse_args()


def read_csv_filtered(csv_path, quality_values, tipos):
    """Read CSV and filter by quality and tipo."""
    quality_set = set(str(q) for q in quality_values)
    tipo_set = set(tipos)
    results = []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        has_tipo_column = reader.fieldnames is not None and 'tipo' in reader.fieldnames
        if not has_tipo_column and tipo_set:
            print("  [INFO] CSV has no 'tipo' column; skipping tipo filter.")
        for row in reader:
            quality_ok = row.get('quality') in quality_set
            tipo_ok = True if not has_tipo_column else row.get('tipo') in tipo_set
            if quality_ok and tipo_ok:
                results.append(row)
    return results


def process_training_images(rows, images_dir, output_dir, num_deg, img_size, overwrite):
    """Generate paired training images with synthetic degradation."""
    source_dir = os.path.join(output_dir, 'source')
    source_mask_dir = os.path.join(output_dir, 'source_mask')

    deg_types = DEGRADATION_TYPES[:num_deg]
    total = len(rows) * len(deg_types)
    processed = 0
    skipped = 0
    errors = 0

    print(f"\n=== Generating training pairs ===")
    print(f"  Clean images: {len(rows)}")
    print(f"  Degradations per image: {len(deg_types)}")
    print(f"  Total pairs to generate: {total}")
    print(f"  Degradation types: {deg_types}\n")

    for i, row in enumerate(rows):
        image_rel = row['image']  # e.g., "patient_folder/uuid.jpg"
        parts = image_rel.replace('\\', '/').split('/')
        if len(parts) < 2:
            print(f"  [WARN] Unexpected path format: {image_rel}")
            errors += 1
            continue

        patient_folder = parts[0]
        image_name = parts[1]
        image_path = os.path.join(images_dir, patient_folder, image_name)

        if not os.path.isfile(image_path):
            print(f"  [WARN] Image not found: {image_path}")
            errors += 1
            continue

        # Load clean image
        try:
            pil_img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"  [WARN] Cannot open {image_path}: {e}")
            errors += 1
            continue

        # Generate mask
        mask_bool = get_mask(pil_img)  # boolean (H, W)

        # Resize mask to target size
        mask_resized = cv2.resize(
            mask_bool.astype(np.uint8), (img_size, img_size),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        for deg_idx, deg_type in enumerate(deg_types):
            flat_name = make_flat_name(patient_folder, image_name, f"_d{deg_idx}")
            source_path = os.path.join(source_dir, flat_name + '.jpg')
            mask_path = os.path.join(source_mask_dir, flat_name + '.png')

            if os.path.exists(source_path) and os.path.exists(mask_path) and not overwrite:
                skipped += 1
                continue

            try:
                degraded_bgr, clean_bgr = apply_degradation(
                    pil_img, mask_bool, img_size, deg_type
                )

                # Concatenate side-by-side: [degraded | clean] → width = 2 * img_size
                paired = np.concatenate([degraded_bgr, clean_bgr], axis=1)
                cv2.imwrite(source_path, paired)

                # Save mask (for the clean image region)
                save_mask_png(mask_resized, mask_path)

                processed += 1
            except Exception as e:
                print(f"  [ERROR] Degradation {deg_type} on {image_rel}: {e}")
                errors += 1
                continue

        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            print(f"  Progress: {i + 1}/{len(rows)} images "
                  f"({processed} pairs created, {skipped} skipped, {errors} errors)")

    print(f"\n  Training data complete: {processed} pairs, {skipped} skipped, {errors} errors")
    return processed


def process_target_images(rows, images_dir, output_dir, img_size, overwrite):
    """Copy target images (for validation/inference) and generate masks."""
    target_dir = os.path.join(output_dir, 'target')
    target_mask_dir = os.path.join(output_dir, 'target_mask')

    processed = 0
    skipped = 0
    errors = 0

    print(f"\n=== Copying target/validation images ===")
    print(f"  Target images: {len(rows)}\n")

    for i, row in enumerate(rows):
        image_rel = row['image']
        parts = image_rel.replace('\\', '/').split('/')
        if len(parts) < 2:
            errors += 1
            continue

        patient_folder = parts[0]
        image_name = parts[1]
        image_path = os.path.join(images_dir, patient_folder, image_name)

        if not os.path.isfile(image_path):
            errors += 1
            continue

        # Use flat naming for target too (consistency)
        flat_name = make_flat_name(patient_folder, image_name)
        target_path = os.path.join(target_dir, flat_name + '.jpg')
        mask_path = os.path.join(target_mask_dir, flat_name + '.png')

        if os.path.exists(target_path) and os.path.exists(mask_path) and not overwrite:
            skipped += 1
            continue

        try:
            pil_img = Image.open(image_path).convert('RGB')

            # Resize to target size and save
            pil_resized = pil_img.resize((img_size, img_size), Image.BICUBIC)
            pil_resized.save(target_path, quality=95)

            # Generate and save mask
            mask_bool = get_mask(pil_img)
            mask_resized = cv2.resize(
                mask_bool.astype(np.uint8), (img_size, img_size),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            save_mask_png(mask_resized, mask_path)

            processed += 1
        except Exception as e:
            print(f"  [ERROR] {image_rel}: {e}")
            errors += 1
            continue

        if (i + 1) % 100 == 0 or (i + 1) == len(rows):
            print(f"  Progress: {i + 1}/{len(rows)} "
                  f"({processed} copied, {skipped} skipped, {errors} errors)")

    print(f"\n  Target data complete: {processed} images, {skipped} skipped, {errors} errors")
    return processed


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    csv_path = os.path.join(str(PROJECT_ROOT), args.csv) if not os.path.isabs(args.csv) else args.csv
    images_dir = os.path.join(str(PROJECT_ROOT), args.images_dir) if not os.path.isabs(args.images_dir) else args.images_dir
    output_dir = os.path.join(str(PROJECT_ROOT), args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    print("=" * 60)
    print("GFE-Net Fine-Tuning Data Preparation")
    print("=" * 60)
    print(f"  CSV:            {csv_path}")
    print(f"  Images dir:     {images_dir}")
    print(f"  Output dir:     {output_dir}")
    print(f"  Image size:     {args.image_size}")
    print(f"  Degradations:   {args.num_degradations}")
    print(f"  Train quality:  {args.quality_train}")
    print(f"  Target quality: {args.quality_target}")
    print(f"  Tipo filter:    {args.tipo}")

    # Create scaffold
    ensure_scaffold(output_dir)

    # Read and filter CSV
    train_rows = read_csv_filtered(csv_path, args.quality_train, args.tipo)
    target_rows = read_csv_filtered(csv_path, args.quality_target, args.tipo)

    print(f"\n  Filtered train (quality={args.quality_train}, tipo={args.tipo}): {len(train_rows)} images")
    print(f"  Filtered target (quality={args.quality_target}, tipo={args.tipo}): {len(target_rows)} images")

    # Generate training pairs
    n_train = process_training_images(
        train_rows, images_dir, output_dir,
        args.num_degradations, args.image_size, args.overwrite
    )

    # Copy target images
    n_target = process_target_images(
        target_rows, images_dir, output_dir,
        args.image_size, args.overwrite
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Training pairs:  {n_train}")
    print(f"  Target images:   {n_target}")
    print(f"  Output dir:      {output_dir}")
    print(f"\n  Directory structure:")
    for sub in ['source', 'source_mask', 'target', 'target_mask']:
        sub_path = os.path.join(output_dir, sub)
        count = len([f for f in os.listdir(sub_path) if os.path.isfile(os.path.join(sub_path, f))]) if os.path.isdir(sub_path) else 0
        print(f"    {sub}/: {count} files")
    print()


if __name__ == '__main__':
    main()
