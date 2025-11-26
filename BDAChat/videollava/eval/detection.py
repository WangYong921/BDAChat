import re
import numpy as np
from tqdm import tqdm
from shapely import wkt
from shapely.wkt import loads
from PIL import Image, ImageDraw
from collections import defaultdict

from videollava.eval.classification import get_string_cleaner, classification_metrics


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.longlong)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-7)
        mAcc = np.nanmean(Acc)
        return mAcc, Acc

    def Pixel_Precision_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Pre = self.confusion_matrix[1, 1] / (self.confusion_matrix[0, 1] + self.confusion_matrix[1, 1])
        return Pre

    def Pixel_Recall_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1])
        return Rec

    def Pixel_F1_score(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.Pixel_Recall_Rate()
        Pre = self.Pixel_Precision_Rate()
        F1 = 2 * Rec * Pre / (Rec + Pre)
        return F1

    def calculate_per_class_metrics(self):
        # Adjustments to exclude class 0 in calculations
        TPs = np.diag(self.confusion_matrix)[1:]  # Start from index 1 to exclude class 0
        FNs = np.sum(self.confusion_matrix, axis=1)[1:] - TPs
        FPs = np.sum(self.confusion_matrix, axis=0)[1:] - TPs
        return TPs, FNs, FPs
    
    def Damage_F1_socore(self):
        TPs, FNs, FPs = self.calculate_per_class_metrics()
        precisions = TPs / (TPs + FPs + 1e-7)
        recalls = TPs / (TPs + FNs + 1e-7)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)
        return f1_scores
    
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int64') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def get_classes(dataset, task):
    if dataset == "OLBDA":
        class_dict = {
            "classification: Classify the level of damage experienced by the building in the second image. Choose from: No damage, Minor Damage, Major Damage, Destroyed.": 
                ["No damage", "Minor damage", "Major damage", "Destroyed"]
        }
    else:
        class_dict = {}
    if task not in class_dict:
        return None
    return class_dict[task]


def detection_metrics(outputs, dataset_name, ignore_casing=True, ignore_punctuation=True):

    task2outputs = defaultdict(list)
    for output in outputs:
        task = output['task']
        task2outputs[task].append(output)

    detection_metrics = {}

    for task in task2outputs:
        if 'OLBDA' in dataset_name:
            if task == 'damage_classification':
                assert dataset_name == 'olbda_qa'
                detection_metrics[f"{task}_accuracy"] = classification_metrics(
                    task2outputs[task],
                    ignore_casing=ignore_casing,
                    ignore_punctuation=ignore_punctuation,
                    keywords=[ "no damage", "minor damage", "major damage", "destroyed"],
                )[f"{task}_accuracy"]

            elif task == 'complex_reasoning':
                assert dataset_name == 'olbda_qa'
                detection_metrics[f"{task}_accuracy"] = classification_metrics(
                    task2outputs[task],
                    ignore_casing=ignore_casing,
                    ignore_punctuation=ignore_punctuation,
                    keywords=[ "no damage", "minor damage", "major damage", "destroyed"],
                )[f"{task}_accuracy"]
            else:
                raise ValueError(f"Unsupported task {task} for dataset {dataset_ame}")

    return detection_metrics
