###################################################################################################
#
# Copyright (C) 2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Classes and functions used to create Color Ball Dataset
Customer custom dataset
"""
import ast
import errno
import xml.etree.ElementTree as ET
from math import ceil
import os
import pickle
import random
import sys

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

import torch
import pandas as pd

from PIL import Image
import ai8x

class BALL(Dataset):

    def __init__(self, root_dir, d_type, transform=None, resize_size=(74, 74)):

        if d_type not in ('test', 'train'):
            raise ValueError("d_type can only be set to 'test' or 'train'")

        if resize_size[0] != resize_size[1]:
            print('resize_size better to be square: Original Images are Square: aspect ratios therefore are not protected')

        self.d_type = d_type
        self.transform = transform
        self.resize_size = resize_size

        self.img_list = list()
        self.boxes_list = list()
        self.lbls_list = list()

        self.raw_data_folder = os.path.join(root_dir, self.__class__.__name__, self.d_type)

        self.processed_folder = os.path.join(root_dir, self.__class__.__name__, 'processed')
        self.__makedir_exist_ok(self.processed_folder)

        res_string = str(self.resize_size[1]) + 'x' + str(self.resize_size[0])

        train_pkl_file_path = os.path.join(self.processed_folder, 'train_' + res_string + '.pkl')
        test_pkl_file_path = os.path.join(self.processed_folder, 'test_' + res_string + '.pkl')


        if self.d_type == 'train':
            self.pkl_file = train_pkl_file_path
        elif self.d_type == 'test':
            self.pkl_file = test_pkl_file_path
        else:
            print(f'Unknown data type: {self.d_type}')
            return

        self.__create_or_read_pkl_file()


    def __create_or_read_pkl_file(self):

        if os.path.exists(self.pkl_file):

            (self.img_list, self.boxes_list, self.lbls_list) = \
                    pickle.load(open(self.pkl_file, 'rb'))
            return
        self.__gen_datasets()

    def __gen_datasets(self):

        print('\nGenerating dataset pickle file from the raw image files...\n')

        img_file_list = sorted([fn for fn in os.listdir(self.raw_data_folder) if fn.endswith(".jpg")])

        total_num_of_processed_files = 0
        for img_file in img_file_list:

            # Construct image file path
            img_file_path = os.path.join(self.raw_data_folder, img_file)

            # Generate corresponding xml file path
            xml_file_path = os.path.join(self.raw_data_folder, f"{img_file[:-4]}.xml")

            # Read image
            image = Image.open(img_file_path)

            # Resize square image:
            img_resized = image.resize(self.resize_size)
            img_resized = np.asarray(img_resized).astype(np.uint8)

            self.img_list.append(img_resized)

            scaling_factor_x = img_resized.shape[0] / image.size[0]
            scaling_factor_y = img_resized.shape[1] / image.size[1]

            # Read XML file and boxes. Each box will have: [x_min,y_min, x_max, y_max] list

            boxes = list()
            lbls = list()

            label_class, original_boxes = BALL.parse_boxes_from_xml_file(xml_file_path)

            for b in range(len(original_boxes)):

                if BALL.check_for_box_validity(original_boxes[b]):

                    # Adjust boxes' coordinates wrt cropped image:
                    x0_new = round(original_boxes[b][0] * scaling_factor_x)
                    y0_new = round(original_boxes[b][1] * scaling_factor_y)
                    x1_new = round(original_boxes[b][2] * scaling_factor_x)
                    y1_new = round(original_boxes[b][3] * scaling_factor_y)

                    boxes.append([x0_new, y0_new, x1_new, y1_new])
                    lbls += label_class[b]
                    
            #print(lbls)
            #print(boxes)
            # All boxes will have label 1
            #lbls = label_class

            self.boxes_list.append(boxes)
            self.lbls_list.append(lbls)

            total_num_of_processed_files = total_num_of_processed_files + 1
        
        # Save pickle file in memory
        pickle.dump((self.img_list, self.boxes_list, self.lbls_list), open(self.pkl_file, 'wb'))

        print(f'\nTotal number of processed files: {total_num_of_processed_files}\n')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        if index >= self.__len__():
            raise IndexError

        if torch.is_tensor(index):
            index = index.tolist()

        img = self.img_list[index]
        boxes = self.boxes_list[index]
        lbls = self.lbls_list[index]

        img = self.__normalize_image(img).astype(np.float32)

        if self.transform is not None:
            img = self.transform(img)

            # Normalize boxes:
            new_boxes = []
            for box in boxes:
                new_boxes.append([box[0] / img.shape[1],
                                  box[1] / img.shape[2],
                                  box[2] / img.shape[1],
                                  box[3] / img.shape[2]])

            boxes = torch.as_tensor(new_boxes, dtype=torch.float32)
            labels = torch.as_tensor(lbls, dtype=torch.int64)

        return img, (boxes, labels)

    @staticmethod
    def collate_fn(batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        images = list()
        boxes_and_labels = list()

        for b in batch:
            images.append(b[0])
            boxes_and_labels.append(b[1])

        images = torch.stack(images, dim=0)
        return images, boxes_and_labels

    @staticmethod
    def parse_boxes_from_xml_file(xml_file_path):

        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        boxes = list()
        label = list()
        
        for object in root.iter('object'):
            name = int(object.find('name').text)
            bbox = object.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            label.append([name])
            boxes.append([xmin, ymin, xmax, ymax])

        return label, boxes

    @staticmethod
    def get_image_size(image_path):
        image = Image.open(image_path)
        return image.size

    @staticmethod
    def __normalize_image(image):
        return image / 256

    @staticmethod
    def __makedir_exist_ok(dirpath):
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

    @staticmethod
    def check_for_box_validity(bbox_x_min_y_min_x_max_y_max):

        x_min = bbox_x_min_y_min_x_max_y_max[0]
        y_min = bbox_x_min_y_min_x_max_y_max[1]
        height = bbox_x_min_y_min_x_max_y_max[2] - x_min
        width = bbox_x_min_y_min_x_max_y_max[3] - y_min

        if height <= 0 or width <= 0 or x_min < 0 or y_min < 0  :
            return False
        return True

def BALL_get_datasets(data, load_train=True, load_test=True, resize_size=(74, 74), load_eggs_only=False):

    """
    """
    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = BALL(root_dir=data_dir, d_type='train',
                                   transform=train_transform, resize_size=resize_size)

        print(f'Train dataset length: {train_dataset.__len__()}\n')
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = BALL(root_dir=data_dir, d_type='test',
                                  transform=test_transform, resize_size=resize_size)

        if args.truncate_testset:
            test_dataset.img_list = test_dataset.img_list[:1]

        print(f'Test dataset length: {test_dataset.__len__()}\n')
    else:
        test_dataset = None

    return train_dataset, test_dataset


def BALL_74_74_get_datasets(data, load_train=True, load_test=True):
    """ """
    return BALL_get_datasets(data, load_train, load_test)

datasets = [

   {
       'name': 'BALL_74_74',
       'input': (3, 74, 74),
       'output': (1, 2, 3, 4),
       'loader': BALL_74_74_get_datasets,
       'collate': BALL.collate_fn
   }

]
