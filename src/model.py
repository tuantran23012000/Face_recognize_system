import os
import cv2
import numpy as np
import warnings
import sys
import argparse
sys.path.append(".")
from Anti_face.anti_spoof_predict import AntiSpoofPredict
from Anti_face.generate_patches import CropImage
from Anti_face.utility import parse_model_name
from Embedding_face import face_preprocess
from Embedding_face.face_model import FaceModel
from Quality_face.test_quality import EQ_face
from Retina_face.detector import RetinaFace
warnings.filterwarnings('ignore')
class KYC():
    def __init__(self,model_type,device):
        ap = argparse.ArgumentParser()
        ap.add_argument('--image-size', default='112,112', help='')
        ap.add_argument('--model_insight', default=os.path.join(os.getcwd(),'pretrained_model/insightface_model/model,0'), 
            help='path to load insightface model')
        ap.add_argument("--model_anti",type=str,default=os.path.join(os.getcwd(),"pretrained_model/anti_spoof_model"),
            help="path to load antiface model")
        ap.add_argument("--model_retina",type=str,default=os.path.join(os.getcwd(),"pretrained_model/retinaface_model/mobilenet0.25_Final.pth"),
            help="path to load retinaface model")
        ap.add_argument("--model_eqface",type=str,default=os.path.join(os.getcwd(),"pretrained_model/eqface_model/Quality_partial_finetuned.pth"),
            help="path to load eqface model")
        ap.add_argument('--backbone', default=os.path.join(os.getcwd(),'pretrained_model/eqface_model/Backbone_glint360k_step3.pth'), type=str, metavar='PATH',
            help='path to backbone eqface model')
        ap.add_argument('--device', default='', type=str,help='gpu/cpu')
        ap.add_argument('--gpu_id', default=0, type=int, help='gpu id')
        ap.add_argument('--cpu_id', default=0, type=int, help='gpu id')
        
        self.args = ap.parse_args()
        self.args.device = device
        self.model_type = model_type
        if self.model_type == "retinaface":
            # Initialize retina model
            self.detector = RetinaFace(self.args) 
        elif self.model_type == "insightface":
            # Initialize insightface model
            self.embedding_model =FaceModel(self.args)
        elif self.model_type == "eqface":
            # Initialize eqface model
            self.quality =EQ_face(self.args)
        elif self.model_type == "antiface":
            # Initialize antiface model
            self.model_test = AntiSpoofPredict(self.args)
        else:
            # Initialize retina model
            self.detector = RetinaFace(self.args) 
            # Initialize insightface model
            self.embedding_model =FaceModel(self.args)
            # Initialize eqface model
            self.quality =EQ_face(self.args)
            # Initialize antiface model
            self.model_test = AntiSpoofPredict(self.args)
        
        
        
    #FUNCTION FACE-ANTI-SPOOFING
    def face_anti_spoofing(self,faces,frame):
        values = []
        labels = []
        for fface in faces:
            bbox = np.array([fface["x1"], fface["y1"], fface["x2"]-fface["x1"]+1, fface["y2"]-fface["y1"]+1])
            image_cropper = CropImage()
            prediction = np.zeros((1, 3))
            # sum the prediction from single model's result
            for model_name in os.listdir(self.args.model_anti):
                h_input, w_input, _, scale = parse_model_name(model_name)
                param = {
                    "org_img": frame,
                    "bbox": bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                if scale is None:
                    param["crop"] = False
                img = image_cropper.crop(**param) 

                prediction += self.model_test.predict(img, os.path.join(self.args.model_anti, model_name))
            # draw result of prediction
            label = np.argmax(prediction)
            value = prediction[0][label]/2
            values.append(value)
            labels.append(label)
        return labels, values

    #FUNCTION RETINA-FACE
    def retinaface_model(self,img):
        faces = []
        face_alls = self.detector(img) 
        for face_all in face_alls:
            ff = {'x1':0, 'y1': 0, 'x2': 0, 'y2': 0,'left_eye':0, 'right_eye': 0, 'nose': 0, 'left_lip': 0, 'right_lip': 0}
            ff["x1"] = int(face_all[0][0])
            ff["y1"] = int(face_all[0][1])
            ff["x2"] = int(face_all[0][2])
            ff["y2"] = int(face_all[0][3])
            ff["left_eye"] = (int(face_all[1][0][0]),int(face_all[1][0][1]))
            ff["right_eye"] = (int(face_all[1][1][0]),int(face_all[1][1][1]))
            ff["nose"] = (int(face_all[1][2][0]),int(face_all[1][2][1]))
            ff["left_lip"] = (int(face_all[1][3][0]),int(face_all[1][3][1]))
            ff["right_lip"] = (int(face_all[1][4][0]),int(face_all[1][4][1]))
            faces.append(ff)
        return faces

    #FUNCTION EQ-FACE
    def eqface_model(self,faces,frame):
        qualities = []
        for fface in faces:
            face = np.array([fface["x1"], fface["y1"], fface["x2"], fface["y2"]])
            keypoints = {"left_eye": fface["left_eye"], "right_eye": fface["right_eye"], 
            "nose": fface["nose"], "left_lip": fface["left_lip"], "right_lip": fface["right_lip"]}
            landmarks = keypoints
            landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["left_lip"][0], landmarks["right_lip"][0],
                landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["left_lip"][1], landmarks["right_lip"][1]])
            landmarks = landmarks.reshape((2,5)).T
            nimg = face_preprocess.preprocess(frame, face, landmarks, image_size='112,112')
            quality = self.quality.predict(nimg)
            qualities.append(quality)
        return qualities

    #FUNCTION INSIGHT-FACE
    def insightface_model(self,faces,frame):
        vector_embeddings = []
        for fface in faces:
            face = np.array([fface["x1"], fface["y1"], fface["x2"], fface["y2"]])
            keypoints = {"left_eye": fface["left_eye"], "right_eye": fface["right_eye"], 
            "nose": fface["nose"], "left_lip": fface["left_lip"], "right_lip": fface["right_lip"]}
            landmarks = keypoints
            landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["left_lip"][0], landmarks["right_lip"][0],
                landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["left_lip"][1], landmarks["right_lip"][1]])
            landmarks = landmarks.reshape((2,5)).T
            nimg = face_preprocess.preprocess(frame, face, landmarks, image_size='112,112')
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            nimg = np.transpose(nimg, (2,0,1))
            #nimg = self.transpose(faces,frame)
            embedding = self.embedding_model.get_feature(nimg)
            vector_embedding = np.array(embedding.reshape(1,512))
            vector_embeddings.append(vector_embedding)
        return vector_embeddings
    def insightface_model1(self,faces,frame):
        vector_embeddings = []
        for fface in faces:
            face = np.array([fface["x1"], fface["y1"], fface["x2"], fface["y2"]])
            keypoints = {"left_eye": fface["left_eye"], "right_eye": fface["right_eye"], 
            "nose": fface["nose"], "left_lip": fface["left_lip"], "right_lip": fface["right_lip"]}
            landmarks = keypoints
            landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["left_lip"][0], landmarks["right_lip"][0],
                landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["left_lip"][1], landmarks["right_lip"][1]])
            landmarks = landmarks.reshape((2,5)).T
            nimg = face_preprocess.preprocess(frame, face, landmarks, image_size='112,112')
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            nimg = np.transpose(nimg, (2,0,1))
            #nimg = self.transpose(faces,frame)
            embedding = self.embedding_model.get_feature(nimg)
            vector_embedding = np.array(embedding.reshape(1,128))
            vector_embeddings.append(vector_embedding)
        return vector_embeddings
    # def insightface_model_1(self,img,face,keypoints):

    #     landmarks = keypoints
    #     landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["left_lip"][0], landmarks["right_lip"][0],
    #                         landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["left_lip"][1], landmarks["right_lip"][1]])
    #     landmarks = landmarks.reshape((2,5)).T
    #     nimg = face_preprocess.preprocess(img, face, landmarks, image_size='112,112')
    #     nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    #     nimg = np.transpose(nimg, (2,0,1))
        
    #     embedding = self.embedding_model.get_feature(nimg).reshape(1,-1)
    #     return embedding
    

    

