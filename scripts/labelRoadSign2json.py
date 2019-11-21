# -*- coding: utf-8 -*-
import argparse
import json
import os
import os.path as osp
import warnings
import PIL.Image
import yaml

import base64

parser = argparse.ArgumentParser()
parser.add_argument('jsonDir')
parser.add_argument('-o', '--out', default=None)


convet2yoloformat = True
rootdir = '../../dataset/roadSign'

args = parser.parse_args()

labelsdir = rootdir + '/labels'


def convert(size, box):
	dw = 1./(size[0])
	dh = 1./(size[1])
	x = (box[0] + box[1])/2.0 - 1
	y = (box[2] + box[3])/2.0 - 1
	w = box[1] - box[0]
	h = box[3] - box[2]
	x = x*dw
	w = w*dw
	y = y*dh
	h = h*dh
	return (x,y,w,h)


def loadJsonfile(jsonFilepath, rootDir):
	label_data = json.load(open(jsonFilepath, 'r'))
	imagePath = label_data['imagePath'].split('..\\')[-1]
	fullPath = os.abspath(rootDir + '/' + imagePath)
	label_shapes = label_data['']


def shapes_to_label(img_shape, shapes, label_name_to_value, type='class'):
	assert type in ['class', 'instance']
	cls = np.zeros(img_shape[:2], dtype=np.int32)
	if type == 'instance':
		ins = np.zeros(img_shape[:2], dtype=np.int32)
		instance_names = ['_background_']
	for shape in shapes:
		points = shape['points']
		label = shape['label']
		shape_type = shape.get('shape_type', None)
		if type == 'class':
		    cls_name = label
		elif type == 'instance':
			cls_name = label.split('-')[0]
			if label not in instance_names:
				instance_names.append(label)
			ins_id = instance_names.index(label)
		cls_id = label_name_to_value[cls_name]
		mask = shape_to_mask(img_shape[:2], points, shape_type)
		cls[mask] = cls_id
		if type == 'instance':
			ins[mask] = ins_id

	if type == 'instance':
		return cls, ins
	return cls

def labelme_shapes_to_label(img_shape, shapes):
	warnings.warn('labelme_shapes_to_label is deprecated, so please use shapes_to_label.')

	label_name_to_value = {'_background_': 0}
	for shape in shapes:
		label_name = shape['label']
		if label_name in label_name_to_value:
			label_value = label_name_to_value[label_name]
		else:
			label_value = len(label_name_to_value)
			label_name_to_value[label_name] = label_value

	lbl = shapes_to_label(img_shape, shapes, label_name_to_value)
	return lbl, label_name_to_value

def convertimgset(img_set="train"):
	if not os.path.exists(labelsdir):
		os.mkdir(labelsdir)
	if convet2yoloformat:
		height = saveimg.shape[0]
		width = saveimg.shape[1]
		txtpath = labelsdir + "/" + filename
		txtpath = txtpath[:-3] + "txt"
		ftxt = open(txtpath, 'w')
		for i in range(len(bboxes)):
			bbox = bboxes[i]
			xcenter = (bbox[0] + bbox[2] * 0.5) / width
			ycenter = (bbox[1] + bbox[3] * 0.5) / height
			wr = bbox[2] * 1.0 / width
			hr = bbox[3] * 1.0 / height
			txtline = "0 " + str(xcenter) + " " + str(ycenter) + " " + str(wr) + " " + str(hr) + "\n"
			ftxt.write(txtline)
		ftxt.close()

 
def main(args):
    warnings.warn("This script is aimed to demonstrate how to convert the\n"
                  "JSON file to a single image dataset, and not to handle\n"
                  "multiple JSON files to generate a real-use dataset.")
    jsonfileDir = args.jsonDir

    # 该段代码在此处无意义
    '''
    if args.out is None:
        out_dir = osp.basename(json_file).replace('.', '_')
        out_dir = osp.join(osp.dirname(json_file), out_dir)
    else:
        out_dir = args.out
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    '''
    list = os.listdir(json_file)
    for i in range(0, len(list)):
        path = os.path.join(json_file, list[i])
        '''
        print('path===================')
        print(path)
        kkk = open(path)
        print(kkk)
        print(type(kkk))
        '''
        if os.path.isfile(path):
            #with open(path,'r') as load_f:  #pang_add method1;
                #data = json.load(load_f)
            #data = json.load(open(path))  #pang_add method2;
            data = json.load(open(path, 'r'))  #pang_add method2;
            '''
            print('data===================')
            print(data)
            print(type(data))
            '''
            img = utils.img_b64_to_array(data['imageData'])
            lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])
 
            captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
            
            lbl_viz = utils.draw_label(lbl, img, captions)
            out_dir = osp.basename(list[i]).replace('.', '_')
            out_dir = osp.join(osp.dirname(list[i]), out_dir)
            if not osp.exists(out_dir):
                os.mkdir(out_dir)
 
            PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
            PIL.Image.fromarray(lbl).save(osp.join(out_dir, 'label.png'))
            PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))
 
            with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
                for lbl_name in lbl_names:
                    f.write(lbl_name + '\n')
 
            warnings.warn('info.yaml is being replaced by label_names.txt')
            info = dict(label_names=lbl_names)
            with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
                yaml.safe_dump(info, f, default_flow_style=False)
 
            print('Saved to: %s' % out_dir)


if __name__ == '__main__':
	main()
