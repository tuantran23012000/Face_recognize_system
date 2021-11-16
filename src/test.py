from torch.autograd import Variable
from anti_spoof_predict import AntiSpoofPredict
import torch.onnx
#import torchvision
import torch
model = AntiSpoofPredict(0)
dummy_input = Variable(torch.randn(1, 3, 128,128))
state_dict = torch.load('./2.7_80x80_MiniFASNetV2.pth',map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
torch.onnx.export(model, dummy_input, "fas.onnx")
