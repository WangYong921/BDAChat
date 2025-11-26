import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import os
import random
import numpy as np
import time
import argparse
import cv2
import copy
import csv
from networks.sam_adapter import build_sam_vit_b_adapter_linknet
from networks.sam_adapter import resize_model_pos_embed
from networks.sam_multi_lora import build_sam_vit_b_adapter_linknet_multi_lora
from networks.sam_lora96_96 import build_sam_vit_b_adapter_linknet_lora96_96
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='dist_b_adapter',
                    help='the name of model')
parser.add_argument('--SAM_pretrained_path', type=str,
                    help='path of SAM_pretrained_weight')
parser.add_argument('--viz_dir', type=str, default='./viz_building/',
                    help='Path to save visualizations and metrics')
parser.add_argument('--val_img_dir', type=str, default=r'\path\to\images',
                    help='Path to validation images')
parser.add_argument('--val_gt_dir', type=str, default=r'\path\to\masks',
                    help='Path to validation ground-truth masks')
parser.add_argument('--image_size', type=int, default=1024, help='image crop size')

args = parser.parse_args()

def evaluate_eng(img_dir, gt_dir, viz_dir, model, img_size=1024):

    val_list = os.listdir(img_dir)
    model = copy.deepcopy(model)
    model.eval()

    os.makedirs(viz_dir, exist_ok=True)
    metrics_csv_path = os.path.join(viz_dir, "metrics.csv")

    metrics_list = []

    for i, name in tqdm(enumerate(val_list), total=len(val_list), desc='Evaluation'):
        if "_post_disaster" in name:
            continue

        img_path = os.path.join(img_dir, name)
        gt_path = os.path.join(gt_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[SKIP] Cannot read image: {img_path}")
            continue
        gt = cv2.imread(gt_path)
        if gt is None:
            print(f"[SKIP] Cannot read GT: {gt_path}")
            continue
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (img_size, img_size), interpolation=cv2.INTER_NEAREST
        img_in = img.transpose(2, 0, 1)
        img_in = torch.from_numpy(img_in.astype(np.float32) / 255.0 * 3.2 - 1.6).unsqueeze(0)

        with torch.no_grad():
            road_output = model(img_in).squeeze().cpu().numpy()
        mask = (road_output > 0.5).astype(np.uint8)
        vis_mask = 255 * mask
        _mask_rgb = np.stack([vis_mask] * 3, axis=2)  # (H, W, 3)
        save_path = os.path.join(viz_dir, name)
        cv2.imwrite(save_path, _mask_rgb)
        gt_bin = (gt == 255).astype(np.uint8)[:, :, 0]

        flat_pred = mask.flatten()
        flat_gt   = gt_bin.flatten()

        try:
            iou = jaccard_score(flat_gt, flat_pred, average='binary')
        except ValueError:
            iou = 0.0
        try:
            precision = precision_score(flat_gt, flat_pred, zero_division=0)
        except ValueError:
            precision = 0.0
        try:
            recall = recall_score(flat_gt, flat_pred, zero_division=0)
        except ValueError:
            recall = 0.0
        try:
            f1 = f1_score(flat_gt, flat_pred, zero_division=0)
        except ValueError:
            f1 = 0.0
        metrics_list.append({
            'image_name': name,
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        })
        print(f"{name} → IoU: {iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    with open(metrics_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['image_name', 'iou', 'precision', 'recall', 'f1_score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics_list:
            writer.writerow(item)

    print(f"\nAll metrics saved to: {metrics_csv_path}")
    return metrics_list

def load(model, path):
    model.load_state_dict(torch.load(path))
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    setup_seed(2333)
    cudnn.benchmark = True
    if args.name == 'b_adapter_sam_sp24':
        model, encoder_global_attn_indexes = build_sam_vit_b_adapter_linknet(
            args.SAM_pretrained_path,
            image_size=args.image_size)
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model.load_state_dict(torch.load(r'\path\to\b_adapter_sam_sp24_60.th'))

    elif args.name == 'b_adapter_sam_multi_lora32_sp24':
        model, encoder_global_attn_indexes = build_sam_vit_b_adapter_linknet_multi_lora(
            args.SAM_pretrained_path, image_size=args.image_size)
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model.load_state_dict(torch.load(r'\path\to\b_adapter_sam_multi_lora32_sp24_60.th'))

    elif args.name == 'b_adapter_sam_lora96_96_sp24':
        model, encoder_global_attn_indexes = build_sam_vit_b_adapter_linknet_lora96_96(
            args.SAM_pretrained_path, image_size=args.image_size)
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model.load_state_dict(torch.load(r'\path\to\b_adapter_sam_lora96_96_sp24_60.th'))

    else:
        raise ValueError(f"Unknown model name: {args.name}")
    log_path = './log_building/' + args.name
    os.makedirs(log_path, exist_ok=True)
    mylog = open(log_path + '/' + args.name + '.log', 'w')

    eng_val_img_dir = args.val_img_dir
    eng_val_gt_dir  = args.val_gt_dir

    metrics_list = evaluate_eng(
        img_dir=eng_val_img_dir,
        gt_dir=eng_val_gt_dir,
        viz_dir=os.path.join(args.viz_dir, args.name + '_results'),
        model=model,
        img_size=args.image_size
    )

    if metrics_list:
        avg_iou = np.mean([m['iou'] for m in metrics_list])
        avg_precision = np.mean([m['precision'] for m in metrics_list])
        avg_recall = np.mean([m['recall'] for m in metrics_list])
        avg_f1 = np.mean([m['f1_score'] for m in metrics_list])

        log_summary = (f"Average IoU: {avg_iou:.4f}, "
                       f"Average Precision: {avg_precision:.4f}, "
                       f"Average Recall: {avg_recall:.4f}, "
                       f"Average F1: {avg_f1:.4f}\n")
        print(log_summary)
        mylog.write(log_summary)

    print('Finish!', file=mylog)
    print('Finish!')
    mylog.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
