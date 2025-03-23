"""
Test script for ResNet model
ResNet 모델을 위한 테스트 스크립트
"""
import torch
import torch.nn as nn
from models.resnet import ResNetModel

def test_resnet(model_size=50, batch_size=2, image_size=224):
    """
    Test a ResNet model with random input
    랜덤 입력으로 ResNet 모델 테스트
    
    Args:
        model_size: ResNet model size (18, 34, 50, 101, 152)
        batch_size: Number of images in batch
        image_size: Size of input images
    """
    print(f"Testing ResNet-{model_size}...")
    print(f"ResNet-{model_size} 테스트 중...")
    
    # Create model | 모델 생성
    resnet = ResNetModel(layers_count=model_size, num_classes=1000)
    model = resnet.get_model()
    
    # Print model summary | 모델 요약 출력
    print(f"\nModel: ResNet-{model_size}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create random input | 랜덤 입력 생성
    dummy_input = torch.randn(batch_size, 3, image_size, image_size)
    
    # Set model to evaluation mode | 모델을 평가 모드로 설정
    model.eval()
    
    # Forward pass | 순전파 수행
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Test completed successfully!\n테스트가 성공적으로 완료되었습니다!")
    return model

if __name__ == "__main__":
    # Test different ResNet variants | 다양한 ResNet 변형 테스트
    for size in [18, 34, 50, 101, 152]:
        test_resnet(size)
        print("-" * 50)
