# MNIST CNN Classifier

PyTorch 기반의 간단한 Convolutional Neural Network(CNN) 모델로 MNIST 손글씨 숫자 데이터셋을 분류합니다.  
이 프로젝트는 CSV 기반의 커스텀 Dataset 클래스를 사용하며, BCELoss 기반의 다중 이진 분류 방식을 따릅니다.

---

## 🧠 Model Structure

```python
Conv2d(1, 10, kernel_size=5, stride=2) → LeakyReLU → BatchNorm  
Conv2d(10, 10, kernel_size=3, stride=2) → LeakyReLU → BatchNorm  
Flatten → Linear(250 → 10) → Sigmoid  
