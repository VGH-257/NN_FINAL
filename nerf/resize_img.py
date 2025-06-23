import os
import imageio
import cv2

input_dir = '/remote-home/xymou/pj/colmap/images'
output_dir = 'fern/images_4'
os.makedirs(output_dir, exist_ok=True)

for i, fname in enumerate(sorted(os.listdir(input_dir))):
    if not fname.endswith(('.png', '.jpg', '.jpeg')):
        continue
    img = imageio.imread(os.path.join(input_dir, fname))
    H, W = img.shape[:2]
    img_small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    imageio.imwrite(os.path.join(output_dir, f'{i:03d}.png'), img_small)