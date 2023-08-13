import os
import cv2
import torch
import numpy as np
import torch.utils.data as data
import xml.etree.ElementTree as ET
from utils import preproc_for_train

VOC_LABELS = (
    'pengiun',
    'turtle'
    )



class VOCDetection(data.Dataset):
    def __init__(self, args, root_path, name='trainval', is_train=True):
        self.is_train = is_train
        self.args = args   
        self.ids = []
        self.root_path = root_path
        
        ano_file = os.path.join(root_path, 'ImageSets', 'Main', name + '.txt')

        with open(ano_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                ano_path = os.path.join(root_path, 'Annotations', line + '.xml')
                img_path = os.path.join(root_path, 'JPEGImages', line + '.jpg')
                self.ids.append((img_path, ano_path))

    def __getitem__(self, index):
        img_path, ano_path = self.ids[index]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        boxes, labels = self.get_annotations(ano_path)
        
        if self.is_train:
            image, boxes, labels = preproc_for_train(image, boxes, labels, self.args.min_size, self.args.mean)
            image = torch.from_numpy(image)
           
        target = np.concatenate([boxes, labels.reshape(-1,1)], axis=1)
        
        return image, target



    def get_annotations(self, path):
        tree = ET.parse(path)

        boxes = []
        labels = []
        
        for child in tree.getroot():
            if child.tag != 'object':
                continue

            bndbox = child.find('bndbox')
            box =[
                float(bndbox.find(t).text) - 1
                for t in ['xmin', 'ymin', 'xmax', 'ymax']
            ]

            label = VOC_LABELS.index(child.find('name').text) 
            
            boxes.append(box)
            labels.append(label)

        return np.array(boxes), np.array(labels)

    
    def __len__(self):
        return len(self.ids)
