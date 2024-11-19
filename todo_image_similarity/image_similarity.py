import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np

# 사전 학습된 ResNet 모델을 사용하여 이미지 임베딩 추출
model = models.resnet50(pretrained=True)
model.eval() # 평가 모드로 전환 (특징 추출을 위해 사용)

# 이미지 전처리 변환
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features.numpy().flatten()

# 이미지 임베딩 추출 예시
image_dir = './images/'  # 이미지 폴더 경로
embeddings = []
image_paths = []

for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.png')):
        path = os.path.join(image_dir, filename)
        features = extract_features(path)
        embeddings.append(features)
        image_paths.append(path)

embeddings = np.array(embeddings)

import faiss

# 임베딩 차원 수
d = embeddings.shape[1]

# FAISS 인덱스 생성
index = faiss.IndexFlatL2(d)  # L2 거리(유클리드 거리)를 기준으로 검색
index.add(embeddings)  # 임베딩 추가

print(f"인덱스에 저장된 벡터 수: {index.ntotal}")

# 검색할 이미지의 특징 벡터 추출
query_image_path = './testimage/vinil.png'  # 예시 이미지 경로
query_features = extract_features(query_image_path).reshape(1, -1)

# 가장 유사한 k개 벡터 검색
k = 3  # 검색할 유사한 이미지 수
distances, indices = index.search(query_features, k)

print(f"Query 이미지: {query_image_path}")
print("유사한 이미지 목록:")
for i in range(k):
    print(f"이미지: {image_paths[indices[0][i]]}, 거리: {distances[0][i]}")