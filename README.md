# ResNet (Residual Networks)

[English](#english) | [한국어](#korean)

<a name="english"></a>
## English

### Overview
This repository contains a PyTorch implementation of ResNet (Residual Networks), a deep convolutional neural network architecture that addresses the degradation problem in deep networks. ResNet introduces skip connections (residual connections) that allow the network to learn residual functions with reference to the layer inputs, rather than learning unreferenced functions.

### Supported Models
- ResNet-18
- ResNet-34
- ResNet-50
- ResNet-101
- ResNet-152

### Architecture
ResNet consists of:
- An initial 7×7 convolution with 64 filters, followed by batch normalization, ReLU, and max pooling
- Four stages of residual blocks
- Global average pooling and a fully-connected layer for classification

ResNet uses two types of building blocks:
1. **Basic Block**: Used in ResNet-18 and ResNet-34, contains two 3×3 convolutions
2. **Bottleneck Block**: Used in ResNet-50, ResNet-101, and ResNet-152, contains 1×1, 3×3, and 1×1 convolutions

### Usage

```python
from models.resnet import ResNetModel

# Create a ResNet-50 model with 1000 classes
resnet = ResNetModel(layers_count=50, num_classes=1000)
model = resnet.get_model()

# For other variants
# ResNet-18: layers_count=18
# ResNet-34: layers_count=34
# ResNet-101: layers_count=101
# ResNet-152: layers_count=152

# Make predictions
import torch
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
```

### Testing
You can easily test the ResNet implementation using the provided test script:

```bash
python test_resnet.py
```

This will create and test all ResNet variants (18, 34, 50, 101, 152) with random input tensors, displaying model information and output shapes.

To test a specific ResNet variant:

```python
# Test specific model
from test_resnet import test_resnet
model = test_resnet(model_size=50)  # Test ResNet-50
```

### Requirements
See `requirements.txt` for the list of dependencies.

### References
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

---

<a name="korean"></a>
## 한국어

### 개요
이 저장소는 ResNet (Residual Networks)의 PyTorch 구현을 포함하고 있습니다. ResNet은 깊은 네트워크에서 발생하는 성능 저하 문제를 해결하는 심층 컨볼루션 신경망 아키텍처입니다. ResNet은 스킵 연결(잔차 연결)을 도입하여 네트워크가 참조되지 않은 함수를 학습하는 대신 레이어 입력에 대한 잔차 함수를 학습할 수 있게 합니다.

### 지원 모델
- ResNet-18
- ResNet-34
- ResNet-50
- ResNet-101
- ResNet-152

### 아키텍처
ResNet은 다음과 같이 구성됩니다:
- 64개 필터를 가진 초기 7×7 컨볼루션, 이후 배치 정규화, ReLU, 최대 풀링
- 4단계의 잔차 블록
- 전역 평균 풀링 및 분류를 위한 완전 연결 레이어

ResNet은 두 가지 유형의 빌딩 블록을 사용합니다:
1. **기본 블록(Basic Block)**: ResNet-18 및 ResNet-34에서 사용되며, 두 개의 3×3 컨볼루션 포함
2. **병목 블록(Bottleneck Block)**: ResNet-50, ResNet-101, ResNet-152에서 사용되며, 1×1, 3×3, 1×1 컨볼루션 포함

### 사용 방법

```python
from models.resnet import ResNetModel

# 1000개 클래스를 가진 ResNet-50 모델 생성
resnet = ResNetModel(layers_count=50, num_classes=1000)
model = resnet.get_model()

# 다른 변형 모델
# ResNet-18: layers_count=18
# ResNet-34: layers_count=34
# ResNet-101: layers_count=101
# ResNet-152: layers_count=152

# 예측하기
import torch
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
```

### 테스트
제공된 테스트 스크립트를 사용하여 ResNet 구현을 쉽게 테스트할 수 있습니다:

```bash
python test_resnet.py
```

이 스크립트는 모든 ResNet 변형(18, 34, 50, 101, 152)을 생성하고 랜덤 입력 텐서로 테스트하여 모델 정보와 출력 형태를 표시합니다.

특정 ResNet 변형을 테스트하려면:

```python
# 특정 모델 테스트
from test_resnet import test_resnet
model = test_resnet(model_size=50)  # ResNet-50 테스트
```

### 요구 사항
의존성 목록은 `requirements.txt`를 참조하세요.

### 참고 문헌
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).