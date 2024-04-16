# MNIST-Classifier-PyTorch

# 프로젝트 이름
PyTorch MNIST Classifier
# 프로젝트 설명
이 프로젝트는 PyTorch를 사용하여 MNIST 데이터셋의 숫자 이미지를 분류하는 심층 신경망 모델을 구현합니다. 목적은 숫자 손글씨를 정확하게 인식하고 분류하는 것입니다.
# 설치 방법
선행 조건: Python 3.x, PyTorch, Pandas, Matplotlib 라이브러리가 설치되어 있어야 합니다.
설치 절차:
bash
Copy code
git clone https://github.com/yourusername/mnist-classifier-pytorch.git
cd mnist-classifier-pytorch
pip install -r requirements.txt
# 사용 방법
훈련 과정:
bash
Copy code
python train.py
테스트 과정:
bash
Copy code
python test.py
# 코드 예제

python
Copy code
from dataset import MnistDataset
from model import Classifier

# 데이터셋 로드
dataset = MnistDataset('path_to_mnist_train.csv')

# 모델 생성 및 훈련
model = Classifier()
model.train(dataset)
기능
MNIST 데이터셋을 사용하여 학습 및 테스트.
모델의 진행 상황 시각화.
테스트 데이터셋에서 모델 성능 평가.
