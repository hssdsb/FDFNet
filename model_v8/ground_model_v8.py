from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from ultralytics import YOLO

import argparse
import collections
import logging
import json
import re
import time
## can be commented if only use LSTM encoder
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
import copy


class grounding_model(nn.Module):
    def __init__(self, bert_model='bert-base-uncased'):
        super(grounding_model, self).__init__()
        if bert_model=='bert-base-uncased':
            self.textdim=768
        else:
            self.textdim=1024

        # visual model
        yolo_model = YOLO('./saved_models/FDFNet_yolov8l.yaml', verbose=True)
        yolo_model = yolo_model.load('./saved_models/yolov8l.pt')
        self.visumodel = yolo_model.model

        # text model
        self.textmodel = BertModel.from_pretrained(bert_model)


    def forward(self, ori_h, ori_w, img_size, img_path, image, word_id, word_mask, bbox):
        batch_size = image.shape[0]
        num_box = bbox.shape[0]

        # Language Module
        all_encoder_layers, _ = self.textmodel(word_id, token_type_ids=None, attention_mask=word_mask)
        # list:12     [tensor(32, 20, 768) tensor(32, 20, 768) ... tensor(32, 20, 768), tensor(32, 20, 768)
        # fix bert during training
        raw_flang = []
        for layer_num in [2, 5, 8, 11]:
            layer = all_encoder_layers[layer_num]
            # layer = layer.detach()
            raw_flang.append(layer)

        layer_final = (all_encoder_layers[-1] + all_encoder_layers[-2] +
                       all_encoder_layers[-3] + all_encoder_layers[-4]) / 4
        # layer_final = layer_final.detach()
        raw_flang.append(layer_final)

        if self.visumodel.training:
            batch = {}
            batch['im_file'] = img_path
            batch['ori_shape'] = [[ori_h[i].item(), ori_w[i].item()] for i in range(len(ori_h))]
            batch['resized_shape'] = [[img_size, img_size] for _ in range(batch_size)]
            batch['img'] = [image, raw_flang, word_mask]
            batch['cls'] = torch.zeros((num_box, 1)).cuda()

            x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            batch['bboxes'] = torch.stack([x_center, y_center, width, height], dim=1).cuda()

            batch['batch_idx'] = torch.arange(batch_size).cuda()

            loss, loss_item = self.visumodel(batch)
            return loss, loss_item
        else:
            batch = [image, raw_flang, word_mask]
            result = self.visumodel(batch)
            return result
