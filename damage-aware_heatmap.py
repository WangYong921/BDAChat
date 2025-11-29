def extract_contours_from_mask(mask_path, min_area=0):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[ERROR] : {mask_path}")
        return []
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
    return filtered

def load_damage_dict_from_json(json_path, num_buildings):

    with open(json_path, 'r') as f:
        responses = json.load(f)

    damage_dict = {}
    for i in range(num_buildings):
        if i < len(responses):

            damage_level = responses[i]["response"].lower().replace(" ", "-")
        else:
            damage_level = "no-damage"

        damage_dict[i] = damage_level
    return damage_dict

def generate_heatmap_overlay_from_contours(
    post_image_path,
    contours,
    output_path,
    json_response_path,
    expand_ratio=0.2,
    blur_kernel=51
):
    post_img = cv2.imread(post_image_path)
    if post_img is None:
        return

    damage_dict = load_damage_dict_from_json(json_response_path, len(contours))

    h_img, w_img = post_img.shape[:2]
    heatmap_gray = np.zeros((h_img, w_img), dtype=np.float32)

    for i, contour in enumerate(contours):
        damage_level = damage_dict.get(i, "no-damage")

        if damage_level == "destroyed":
            intensity = 1.0
        elif damage_level == "major-damage":
            intensity = 0.4
        elif damage_level == "minor-damage":
            intensity = 0.2
        else:
            continue

        x, y, w_box, h_box = cv2.boundingRect(contour)
        pad_x = int(w_box * expand_ratio)
        pad_y = int(h_box * expand_ratio)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w_box + pad_x)
        y2 = min(h_img, y + h_box + pad_y)
        expanded_box = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ]).reshape((-1, 1, 2))


        cv2.drawContours(heatmap_gray, [expanded_box], -1, intensity, thickness=cv2.FILLED)


    heatmap_blurred = cv2.GaussianBlur(heatmap_gray, (blur_kernel, blur_kernel), 0)
    heatmap_blurred = np.clip(heatmap_blurred, 0, 1)
    heatmap_color = cv2.applyColorMap((heatmap_blurred * 255).astype(np.uint8), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(post_img, 0.5, heatmap_color, 0.5, 0)
    cv2.imwrite(output_path, overlay)
    print(f"[INFO] Heatmap overlay image saved successfully：{output_path}")

for block_id in range(1, 100):#number of block
    post_image_path = rf"\path\to\block\images\block_{block_id}_post_disaster.png"
    mask_path = rf"\path\to\block\masks\block_{block_id}_pre_disaster.png"
    output_dir = rf"\path\to\output_name\output{block_id}"
    output_path = os.path.join(output_dir, "heatmap_overlay.png")
    json_response_path = rf"bdachat-7B_prompt_strategy_interleave_chronological_prefix_True{block_id}.json"

    print(f"\n{'='*50}")
    print(f"[INFO] start block_{block_id}")
    print(f"[INFO] post_image_path: {post_image_path}")
    print(f"[INFO] mask_path: {mask_path}")
    print(f"[INFO] output_path: {output_path}")
    print(f"[INFO] json_response_path: {json_response_path}")
    print(f"{'='*50}")

    os.makedirs(output_dir, exist_ok=True)

    contours = extract_contours_from_mask(mask_path)
    if not contours:

        continue

    generate_heatmap_overlay_from_contours(
        post_image_path=post_image_path,
        contours=contours,
        output_path=output_path,
        json_response_path=json_response_path,
        expand_ratio=0,
        blur_kernel=85
    )

print(f"\n[INFO] finish！")