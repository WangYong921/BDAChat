import os
import cv2
import numpy as np

def extract_buildings_pre_post(
    pre_image_path, post_image_path, pre_mask_path, output_dir,
    use_rotated_rect=False, visualize=False, expand_ratio=0.1
):
    os.makedirs(output_dir, exist_ok=True)

    pre_img = cv2.imread(pre_image_path)
    post_img = cv2.imread(post_image_path)
    mask = cv2.imread(pre_mask_path, cv2.IMREAD_GRAYSCALE)

    if pre_img is None or post_img is None or mask is None:
        print(f"[ERROR] ")
        return

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = pre_img.shape[:2]

    valid_count = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 0:
            continue

        if use_rotated_rect:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            x, y, w, h = cv2.boundingRect(box)
        else:
            x, y, w, h = cv2.boundingRect(contour)

        pad_w = int(w * expand_ratio)
        pad_h = int(h * expand_ratio)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w_img, x + w + pad_w)
        y2 = min(h_img, y + h + pad_h)

        pre_crop = pre_img[y1:y2, x1:x2]
        post_crop = post_img[y1:y2, x1:x2]

        cv2.imwrite(os.path.join(output_dir, f"building_{valid_count}_pre.png"), pre_crop)
        cv2.imwrite(os.path.join(output_dir, f"building_{valid_count}_post.png"), post_crop)
        valid_count += 1

    if visualize:
        vis = pre_img.copy()
        valid_count = 0
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            pad_w = int(w * expand_ratio)
            pad_h = int(h * expand_ratio)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w_img, x + w + pad_w)
            y2 = min(h_img, y + h + pad_h)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{valid_count}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            valid_count += 1

        cv2.imwrite(os.path.join(output_dir, "buildings_overlay_pre.png"), vis)

    print(f"[INFO] Extraction complete. A total of {len(contours)} building instances saved.")

for block_id in range(1, 100):#number of block
    pre_image_path = rf"\path\to\block\images\block_{block_id}_pre_disaster.png"
    post_image_path = rf"\path\to\block\images\block_{block_id}_post_disaster.png"
    pre_mask_path = rf"\path\to\block\masks\block_{block_id}_pre_disaster.png"
    output_dir = rf"\path\to\output_name\output{block_id}"

    print(f"\n{'='*50}")
    print(f"[INFO] start block_{block_id}")
    print(f"[INFO] outputdir: {output_dir}")
    print(f"{'='*50}")


    extract_buildings_pre_post(
        pre_image_path=pre_image_path,
        post_image_path=post_image_path,
        pre_mask_path=pre_mask_path,
        output_dir=output_dir,
        use_rotated_rect=True,
        visualize=True,
        expand_ratio=0.8
    )

print(f"\n[INFO] finish！")