import torch
import cv2
import os

from Quality_face.MobileFaceNets import MobileFaceNet
from Quality_face.EQFace import FaceQuality

import argparse
import numpy as np

class EQ_face():
    def __init__(self,args):
        print(os.getcwd())
        # parser = argparse.ArgumentParser(description='PyTorch Face Quality test')
        # parser.add_argument('--backbone', default=os.path.join(os.getcwd(),'eq_face/Backbone_glint360k_step3.pth'), type=str, metavar='PATH',
        #                     help='path to backbone model')
        # parser.add_argument('--gpu', default=0, type=int,
        #                     help='index of gpu to run')
        self.args = args
        self.model = self.args.model_eqface
        if self.args.device == "gpu":
            self.device_id= self.args.gpu_id
        else:
            self.device_id = -1
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.gpu)
        # self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.DEVICE = (
            torch.device("cpu") if self.device_id == -1 else torch.device("cuda", self.device_id)
        )
        self.BACKBONE = MobileFaceNet(512,7,7)
        self.QUALITY = FaceQuality(512)
        if os.path.isfile(self.args.backbone):
            print("Loading Backbone Checkpoint '{}'".format(self.args.backbone))
            checkpoint = torch.load(self.args.backbone, map_location='cpu')
            self.load_state_dict(self.BACKBONE, checkpoint)
        else:
            print("No Checkpoint Found at '{}' Please Have a Check or Continue to Train from Scratch".format(self.args.backbone))
            return
        if os.path.isfile(self.model):
            print("Loading Quality Checkpoint '{}'".format(self.model))
            checkpoint = torch.load(self.model, map_location='cpu')
            self.load_state_dict(self.QUALITY, checkpoint)
        else:
            print("No Checkpoint Found at '{}' Please Have a Check or Continue to Train from Scratch".format(self.model))
            return
        self.BACKBONE.to(self.DEVICE)
        self.QUALITY.to(self.DEVICE)
        self.BACKBONE.eval()
        self.QUALITY.eval()
    def load_state_dict(self,model, state_dict):
        all_keys = {k for k in state_dict.keys()}
        for k in all_keys:
            if k.startswith('module.'):
                state_dict[k[7:]] = state_dict.pop(k)
        model_dict = model.state_dict()
        pretrained_dict = {k:v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        if len(pretrained_dict) == len(model_dict):
            print("all params loaded")
        else:
            not_loaded_keys = {k for k in pretrained_dict.keys() if k not in model_dict.keys()}
            print("not loaded keys:", not_loaded_keys)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    def get_face_quality(self,backbone, quality, device, img):
        resized = cv2.resize(img, (112, 112))
        ccropped = resized[...,::-1] # BGR to RGB
        # load numpy to tensor
        ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
        ccropped = np.reshape(ccropped, [1, 3, 112, 112])
        ccropped = np.array(ccropped, dtype = np.float32)
        ccropped = (ccropped - 127.5) / 128.0
        ccropped = torch.from_numpy(ccropped)

        # extract features
        backbone.eval() # set to evaluation mode
        with torch.no_grad():
            _, fc = backbone(ccropped.to(device), True)
            s = quality(fc)[0]
        #print(s)
        return s.cpu().numpy()

    def predict(self,frame):
        
        #frame = cv2.imread(image)
        if frame is None or frame.shape[0] == 0:
            print("Open image failed: ")
            return
        quality = self.get_face_quality(self.BACKBONE, self.QUALITY, self.DEVICE, frame)
        return quality


