import argparse
import json
import cv2
__author__ = 'yuqj'
__copyright__ = 'Copyright (c) 2018, deepano'
__email__ = 'yuqj@deepano.com'
__license__ = 'DEEPANO'


anno_root = "../../dataset/car_person_data/bdd100k/Annotations/"
label_train_root = "../../dataset/car_person_data/bdd100k/labels/train/"
label_test_root = "../../dataset/car_person_data/bdd100k/labels/test/"
anno_image = "../../dataset/car_person_data/bdd100k/annoImage/"
src_image_train_root = "../../dataset/car_person_data/bdd100k/JPEGImages/100k/train/"
src_image_val_root = "../../dataset/car_person_data/bdd100k/JPEGImages/100k/val/"
src_image_test_root = "../../dataset/car_person_data/bdd100k/JPEGImages/100k/test/"
label_json_file = ["../../dataset/car_person_data/bdd100k/labels/val/bdd100k_labels_images_train.json",
                   "../../dataset/car_person_data/bdd100k/labels/val/bdd100k_labels_images_val.json"]
src_image_root = [src_image_train_root, src_image_val_root]
category_label = ['parking sign', 'street light', 'traffic cone', 'traffic device', 'traffic light',
                  'traffic sign', 'person', 'rider', 'bicycle', 'bus', 'car', 'caravan', 'motorcycle', 'trailer',
                  'train', 'truck']


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('label_path', type=str, help="this should be label json file")
    parse.add_argument('det_path', type=str, help="this should be label or annotation file path")


def label2det(frames, src_image_, label_):
    boxes = list()
    for frame in frames:
        image_file = frame['name']
        label_file = label_ + image_file.split('.')[0]+'.txt'
        anno_xml_file = anno_root + frame['name']
        anno_image_file = anno_image + frame['name']
        src_image_file = src_image_ + frame['name']
        if 1:
            print("anno_xml_file: ", anno_xml_file)
            print("anno_image_file: ", anno_image_file)
            print("src_image_file: ", src_image_file)
        srcImage = cv2.imread(src_image_file)
        label_w_file = open(label_file, 'w')
        for label in frame['labels']:
            if 'box2d' not in label:
                continue
            category_index = 0
            if label['category'] not in category_label:
                continue
            else:
                for ii in range(len(category_label)):
                    if label['category'] == category_label[ii]:
                        category_index = ii
                        break
            xy = label['box2d']
            if xy['x1'] >= xy['x2'] or xy['y1'] >= xy['y2']:
                continue
            box = {'name': frame['name'],
                   'timestamp': frame['timestamp'],
                   'category': label['category'],
                   'bbox': [xy['x1'], xy['y1'], xy['x2'], xy['y2']],
                   'score': 1}
            category = label["category"]
            x1 = xy['x1']
            y1 = xy['y1']
            x2 = xy['x2']
            y2 = xy['y2']
            # labels.txt
            label_content = category_index + ' ' + x1 + ' ' + x2 + ' ' + y1 + ' ' + y2 +'\n'
            label_w_file.write(label_content)
            # anno image
            cv2.rectangle(srcImage, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0))
            cv2.putText(srcImage, category, (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, 6, (0, 0, 255), 25)
            # anno_xml_file
            
            boxes.append(box)
        cv2.imwrite(anno_image_file, srcImage)
        label_w_file.close()
    return boxes


def convert_labels(label_path, det_path):
    frames = json.load(open(label_path, 'r'))
    det = label2det(frames)