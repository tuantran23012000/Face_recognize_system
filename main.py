from src.model import KYC
import cv2

model_type = ""
device = "gpu"
kyc = KYC(model_type,device)
frame = cv2.imread("test2.png")
#frame = cv2.imread("E:/AI/FTECH/CODE/EKYC/eKYC/img_test2.jpg")

# Module Retinaface (Face detection)
faces = kyc.retinaface_model(frame) #without batch
print(faces)

# Module Antiface (Face anti-spoofing)
labels, values = kyc.face_anti_spoofing(faces,frame) #without batch
print(labels, values)

# Module Insightface (Face embedding)
vectors = kyc.insightface_model(faces,frame) #512-D without batch
print(vectors)

# Module EQface (Face quality)
qualities = kyc.eqface_model(faces,frame) #512-D without batch
print(qualities)