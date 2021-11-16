import os

import numpy as np
import torch

from .alignment import load_net, batch_detect


def get_project_dir():
    current_path = os.path.abspath(os.path.join(__file__, "../"))
    return current_path


def relative(path):
    path = os.path.join(get_project_dir(), path)
    return os.path.abspath(path)


class RetinaFace:
    def __init__(self,args):
        network="mobilenet"
        self.args = args
        if self.args.device == "gpu":
            self.device_id= self.args.gpu_id
        else:
            self.device_id = -1
        
        self.device_model = (
            torch.device("cpu") if self.device_id == -1 else torch.device("cuda", self.device_id)
        )
        self.model = load_net(self.args.model_retina, self.device_model, network)

    def detect(self, images):
        if isinstance(images, np.ndarray):
            if len(images.shape) == 3:
                return batch_detect(self.model, [images], self.device_model)[0]
            elif len(images.shape) == 4:
                return batch_detect(self.model, images, self.device_model)
        elif isinstance(images, list):
            return batch_detect(self.model, np.array(images), self.device_model)
        elif isinstance(images, torch.Tensor):
            if len(images.shape) == 3:
                return batch_detect(self.model, images.unsqueeze(0), self.device_model)[0]
            elif len(images.shape) == 4:
                return batch_detect(self.model, images, self.device_model)
        else:
            raise NotImplementedError()

    def __call__(self, images):
        return self.detect(images)
