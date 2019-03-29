import argparse
import json
import cv2
__author__ = 'yuqj'
__copyright__ = 'Copyright (c) 2018, deepano'
__email__ = 'yuqj@deepano.com'
__license__ = 'DEEPANO'


anno_root = "../../dataset/car_person_data/bdd100k/Annotations/"
anno_image = "../../dataset/car_person_data/bdd100k/annoImage/"
src_image_train_root = "../../dataset/car_person_data/bdd100k/JPEGImages/100k/train/"
src_image_val_root = "../../dataset/car_person_data/bdd100k/JPEGImages/100k/val/"
src_image_test_root = "../../dataset/car_person_data/bdd100k/JPEGImages/100k/test/"
label_json_file = ["../../dataset/car_person_data/bdd100k/labels/val/bdd100k_labels_images_train.json",
                   "../../dataset/car_person_data/bdd100k/labels/val/bdd100k_labels_images_val.json"]
src_image_root =[src_image_train_root, src_image_val_root]


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('label_path', type=str, help="this should be label json file")
    parse.add_argument('det_path', type=str, help="this should be label or annotation file path")


def label2det(frames, src_image_root):
    boxes = list()
    for frame in frames:
        for label in frame['labels']:
            if 'box2d' not in label:
                continue
            xy = label['box2d']
            if xy['x1'] >= xy['x2'] or xy['y1'] >= xy['y2']:
                continue
            anno_file = anno_root+frame['name']
            anno_image_file = anno_image+frame['name']
            src_image_file = src_image_root + 
            box = {'name': frame['name'],
                   'timestamp': frame['timestamp'],
                   'category': label['category'],
                   'bbox': [xy['x1'], xy['y1'], xy['x2'], xy['y2']],
                   'score': 1}
            boxes.append(box)
    return boxes


def convert_labels(label_path, det_path):
    frames = json.load(open(label_path, 'r'))
    det = label2det(frames)