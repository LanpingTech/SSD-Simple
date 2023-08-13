import os
import shutil
import cv2
import numpy as np
import os
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import random
import json

def save_xml(image_name, target, save_dir, width=1080, height=1920, channel=3):
    # 生成xml文件
    '''
    :param image_name: 图片名
    :param bbox: 对应的bbox
    :param save_dir: xml文件保存路径
    :param width: 图片宽度
    :param height: 图片高度
    :param channel: 图片通道
    :return:
    '''
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel

    for label, xmin, ymin, xmax, ymax in target: # bbox是个set, 内容类似于{(1,2,3,5),(5,6,2,5)}
        node_object = SubElement(node_root, 'object')

        node_name = SubElement(node_object, 'name')
        node_name.text = label

        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        node_bndbox = SubElement(node_object, 'bndbox')

        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % xmin

        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % ymin

        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % xmax

        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % ymax

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)

    save_xml = os.path.join(save_dir, image_name.replace('jpg', 'xml'))

    with open(save_xml, 'wb') as f: # 自动关闭文件，无需手动书写close()
        f.write(xml)

    return

def int2labelstr(label):
    if label == 1:
        return 'pengiun'
    elif label == 2:
        return 'turtle'
    else:
        return 'unknown'

if os.path.exists('dataset'):
    shutil.rmtree('dataset')

os.makedirs('dataset')
os.makedirs('dataset/Annotations')
os.makedirs('dataset/ImageSets')
os.makedirs('dataset/ImageSets/Main')
os.makedirs('dataset/JPEGImages')

ori_data = 'data'
ori_train_imgs_path = os.path.join(ori_data, 'train')
ori_train_anno = os.path.join(ori_data, 'train_annotations')
ori_val_data = os.path.join(ori_data, 'valid')
ori_val_anno = os.path.join(ori_data, 'valid_annotations')

train_list = []
train_annotations = json.load(open(ori_train_anno))
for anno in train_annotations:
    old_img_name = 'image_id_{}.jpg'.format(str(anno['image_id']).zfill(3))
    img_path = os.path.join(ori_train_imgs_path, old_img_name)
    new_img_name = 'train_{}.jpg'.format(str(anno['image_id']).zfill(3))
    shutil.copy(img_path, os.path.join('dataset/JPEGImages', new_img_name))

    bbox = anno['bbox']
    new_bbox = []
    new_bbox.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])

    label = anno['category_id']
    new_label = []
    new_label.append(int2labelstr(label))

    target = []
    for bbox, label in zip(new_bbox, new_label):
        target.append([label, bbox[0], bbox[1], bbox[2], bbox[3]])

    save_xml(new_img_name, target, 'dataset/Annotations')

    train_list.append(new_img_name.replace('.jpg', ''))

val_list = []
val_annotations = json.load(open(ori_val_anno))
for anno in val_annotations:
    old_img_name = 'image_id_{}.jpg'.format(str(anno['image_id']).zfill(3))
    img_path = os.path.join(ori_val_data, old_img_name)
    new_img_name = 'val_{}.jpg'.format(str(anno['image_id']).zfill(3))
    shutil.copy(img_path, os.path.join('dataset/JPEGImages', new_img_name))

    bbox = anno['bbox']
    new_bbox = []
    new_bbox.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])

    label = anno['category_id']
    new_label = []
    new_label.append(int2labelstr(label))

    target = []
    for bbox, label in zip(new_bbox, new_label):
        target.append([label, bbox[0], bbox[1], bbox[2], bbox[3]])

    save_xml(new_img_name, target, 'dataset/Annotations')

    val_list.append(new_img_name.replace('.jpg', ''))

trainval_list = train_list + val_list
# write trainval.txt
with open('dataset/ImageSets/Main/trainval.txt', 'w') as f:
    for name in trainval_list:
        f.write(name + '\n')

# write train.txt
with open('dataset/ImageSets/Main/train.txt', 'w') as f:
    for name in train_list:
        f.write(name + '\n')

# write val.txt
with open('dataset/ImageSets/Main/val.txt', 'w') as f:
    for name in val_list:
        f.write(name + '\n')

# write test.txt
with open('dataset/ImageSets/Main/test.txt', 'w') as f:
    for name in val_list:
        f.write(name + '\n')

# write voc.names
with open('dataset/voc.names', 'w') as f:
    f.write('pengiun\n')
    f.write('turtle\n')


    



