import os
import torch
import logging
import torch.nn as nn
import importlib
from model.clip.model import build_model
from utils import get_logger, get_summary_writer

class Pre_Layer(nn.Module):
    def __init__(self, inputdim=2048, nb_class=64):
        super(Pre_Layer, self).__init__()
        self.fc = nn.Linear(inputdim, nb_class)

    def forward(self, data):
        pre = self.fc(data)
        return pre

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class LinearHash(nn.Module):

    def __init__(self, inputDim=2048, outputDim=64):
        super(LinearHash, self).__init__()
        self.fc = nn.Linear(inputDim, outputDim)
        self.fc.apply(weights_init_kaiming)
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, data):
        result = self.fc(data)
        return torch.tanh(self.drop_out(result))

class BaseBackbone(nn.Module):

    def __init__(self,
                 outputDim=64,
                 preload=None,
                 writer=None,
                 saveDir="./result/log",
                 logger: logging.Logger = None,
                 is_train=True):
        super(BaseBackbone, self).__init__()

        self.embedDim, self.backbone = self.load_model(preload)

        os.makedirs(saveDir, exist_ok=True)
        self.logger = logger if logger is not None else get_logger(
            os.path.join(saveDir, "train.log" if is_train else "test.log"))
        self.writer = writer if writer is not None and is_train else get_summary_writer(
            os.path.join(saveDir, "tensorboard"))

        if type(self.embedDim) == list:
            self.image_hash = LinearHash(inputDim=self.embedDim[0], outputDim=outputDim)
            self.text_hash = LinearHash(inputDim=self.embedDim[1], outputDim=outputDim)
        else:
            self.image_hash = LinearHash(inputDim=self.embedDim, outputDim=outputDim)
            self.text_hash = LinearHash(inputDim=self.embedDim, outputDim=outputDim)

    def load_model(self, preload):
        module = importlib.import_module(model.clip)
        return module.load_backbone(preload=preload)

    def encode_image(self, image):

        image_embed = self.backbone.encode_image(image)  # 512
        image_embed = self.image_hash(image_embed)
        return image_embed

    def eval(self):
        self.image_hash.eval()
        self.text_hash.eval()

    def train(self):
        self.image_hash.train()
        self.text_hash.train()

    def encode_text(self, text):

        text_embed = self.backbone.encode_text(text)
        text_embed = self.text_hash(text_embed)

        return text_embed

    def forward(self, image, text):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)
        return image_embed, text_embed