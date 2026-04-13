# -*- coding: UTF-8 -*-
"""
@Function: 给source和target的做一个mask
@File: get_mask.py
@Date: 2021/6/10 20:50
@Author: Hever
"""
import os
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from optparse import OptionParser


def get_mask(img, ideal_mask=None):
    gray = np.array(img.convert('L'))
    mask = ndimage.binary_opening(gray > 10, structure=np.ones((8, 8)))
    
    if ideal_mask is not None:
        if ideal_mask.shape != mask.shape:
            ideal_mask_resized = cv2.resize(ideal_mask.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            ideal_mask_bool = ideal_mask_resized > 0
        else:
            ideal_mask_bool = ideal_mask > 0
        
        mask = np.logical_or(mask, ideal_mask_bool)
        
    return mask


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--image_dir', default='./images/drive_cataract/source',
                      help="input directory to the source image.")
    parser.add_option('--output_dir', default='./images/drive_cataract/source_mask',
                      help="output directory to the source mask.")
    parser.add_option('--mode', default='pair',
                      help="pair option is for the source image, single option is for the target image")
    parser.add_option('--ideal_mask', default=None,
                      help="path to an ideal mask image to fill missing areas.")
    (opt, args) = parser.parse_args()
    image_dir = opt.image_dir
    output_dir = opt.output_dir

    ideal_mask_img = None
    if opt.ideal_mask and os.path.exists(opt.ideal_mask):
        ideal_mask_img = cv2.imread(opt.ideal_mask, cv2.IMREAD_GRAYSCALE)

    mkdir(output_dir)
    for root, dirs, files in os.walk(image_dir):
        for image_name in files:
            # Pula casos de arquivos escondidos e não-imagens para evitar novos erros
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
                
            image_path = os.path.join(root, image_name)
            output_path_A = os.path.join(output_dir, image_name.split('.')[0] + '.png')

            if opt.mode == 'pair':
                SAB = Image.open(image_path).convert('RGB')
                w, h = SAB.size
                w2 = int(w / 2)
                SA = SAB.crop((0, 0, w2, h))
                image = SAB.crop((w2, 0, w, h))
            else:
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception as e:
                    print(f"Erro ao abrir a imagem {image_path}: {e}")
                    continue

            mask = get_mask(image, ideal_mask=ideal_mask_img)
            cv2.imwrite(output_path_A, mask * 255)