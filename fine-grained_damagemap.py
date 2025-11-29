def visualize_damage_on_image(
    original_image_path,
    contours,
    output_mask_path,
    json_response_path
):

    image = cv2.imread(original_image_path)
    if image is None:
        print(f"[ERROR] ：{original_image_path}")
        return

    with open(json_response_path, 'r') as f:
        responses = json.load(f)

    label_map = image.copy()


    valid_count = 0

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 0:
            continue

        if valid_count < len(responses):
            damage_label = responses[valid_count]["response"]
        else:
            damage_label = "No damage"  


        if damage_label == "Destroyed":
            color_label = (0, 0, 255)   # Red
        elif damage_label == "Major Damage":
            color_label = (0, 165, 255)  # Orange
        elif damage_label == "Minor Damage":
            color_label = (0, 255, 255)  # Yellow
        else:  # No damage
            color_label = (0, 255, 0)   # Green


        cv2.drawContours(label_map, [contour], -1, color_label, thickness=cv2.FILLED)

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(label_map, f"{valid_count}", (cX-10, cY),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        valid_count += 1

    cv2.imwrite(output_mask_path, label_map)
    print(f"[INFO] Has saved the annotated image to {output_mask_path}，with a total of{valid_count}buildings")

base_img_dir = r"\path\to\block\images"
base_mask_dir = r"\path\to\block\masks"
base_output_dir = r"\path\to\output_name"
json_template = r"bdachat-7B_prompt_strategy_interleave_chronological_prefix_True{}.json"

for block_id in range(1, 100):#number of block
    img_path = os.path.join(base_img_dir, f"block_{block_id}_post_disaster.png")
    mask_path = os.path.join(base_mask_dir, f"block_{block_id}_pre_disaster.png")
    output_dir = os.path.join(base_output_dir, f"output{block_id}")
    output_mask = os.path.join(output_dir, "damage_overlay.png")
    json_response_path = json_template.format(block_id)


    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"[INFO] start block_{block_id}")
    print(f"[INFO] image_path: {img_path}")
    print(f"[INFO] mask_path: {mask_path}")
    print(f"[INFO] JSON_path: {json_response_path}")
    print(f"[INFO] output_path: {output_mask}")
    print(f"{'='*50}")

    missing_files = []
    if not os.path.exists(img_path):
        missing_files.append(f"img_path: {img_path}")
    if not os.path.exists(mask_path):
        missing_files.append(f"mask_path: {mask_path}")
    if not os.path.exists(json_response_path):
        missing_files.append(f"JSON_path: {json_response_path}")

    if missing_files:
        print(f"[WARNING] ")
        for file in missing_files:
            print(f"  - {file}")
        continue


    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[INFO] find {len(contours)} buildings")


    visualize_damage_on_image(
        original_image_path=img_path,
        contours=contours,
        output_mask_path=output_mask,
        json_response_path=json_response_path,
        expand_ratio=0.2
    )

print(f"\n[INFO] all block finish！")