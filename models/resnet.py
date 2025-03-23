import torch
from torch import nn

class BasicBlock(nn.Module):
    """
    Basic building block for ResNet-18/34 variants
    ResNet-18/34 변형에 사용되는 기본 구성 요소
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, projection = None):
        """
        Initialize a basic block
        기본 블록 초기화
        
        Args:
            in_channels: Number of input channels | 입력 채널 수
            out_channels: Number of output channels | 출력 채널 수
            stride: Stride for convolution | 컨볼루션의 스트라이드
            projection: Projection for shortcut connection if dimensions don't match | 차원이 일치하지 않을 때 사용할 단축 연결 투영
        """
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        self.projection = projection
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass for the basic block
        기본 블록의 순전파
        """
        residual = self.residual(x)
        if self.projection is not None:
            shortcut = self.projection(x)
        else:
            shortcut = x
        res = self.relu(residual + shortcut)
        return res

class BottleNeck(nn.Module):
    """
    Bottleneck building block for ResNet-50/101/152 variants
    ResNet-50/101/152 변형에 사용되는 병목 구성 요소
    """
    expansion = 4
    def __init__(self, in_channels, inner_channels, stride=1, projection = None):
        """
        Initialize a bottleneck block
        병목 블록 초기화
        
        Args:
            in_channels: Number of input channels | 입력 채널 수
            inner_channels: Number of intermediate channels | 중간 채널 수
            stride: Stride for convolution | 컨볼루션의 스트라이드
            projection: Projection for shortcut connection if dimensions don't match | 차원이 일치하지 않을 때 사용할 단축 연결 투영
        """
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels * self.expansion)
        )
        self.projection = projection
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass for the bottleneck block
        병목 블록의 순전파
        """
        residual = self.residual(x)
        if self.projection is not None:
            shortcut = self.projection(x)
        else:
            shortcut = x
        res = self.relu(residual + shortcut)
        return res


class ResNet(nn.Module):
    """
    Implementation of ResNet architecture
    ResNet 아키텍처 구현
    """
    def __init__(self, block, num_block_list, num_classes=1000, zero_init_residual=True):
        """
        Initialize ResNet model
        ResNet 모델 초기화
        
        Args:
            block: Block type (BasicBlock or BottleNeck) | 블록 유형 (BasicBlock 또는 BottleNeck)
            num_block_list: List with number of blocks per stage | 스테이지당 블록 수를 포함한 리스트
            num_classes: Number of output classes | 출력 클래스 수
            zero_init_residual: Whether to initialize residual branch with zeros | 잔차 분기를 0으로 초기화할지 여부
        """
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self.make_stage(block, 64, num_block_list[0], 1)
        self.stage2 = self.make_stage(block, 128, num_block_list[1], 2)
        self.stage3 = self.make_stage(block, 256, num_block_list[2], 2)
        self.stage4 = self.make_stage(block, 512, num_block_list[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
        # Initialize weights for convolutional layers | 컨볼루션 레이어의 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # Zero-initialize the last BN in each residual branch | 각 잔차 분기의 마지막 BN 레이어를 0으로 초기화
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, block):
                    nn.init.constant_(m.residual[-1].weight, 0)

    def make_stage(self, block, inner_channels, num_blocks, stride=1):
        """
        Create a stage with multiple blocks
        여러 블록으로 구성된 스테이지 생성
        
        Args:
            block: Block type | 블록 유형
            inner_channels: Number of intermediate channels | 중간 채널 수
            num_blocks: Number of blocks in this stage | 이 스테이지의 블록 수
            stride: Stride for the first block | 첫 번째 블록의 스트라이드
        
        Returns:
            Sequential container with blocks | 블록을 포함하는 Sequential 컨테이너
        """
        if stride != 1 or self.in_channels != inner_channels * block.expansion:
            projection = nn.Sequential(
                nn.Conv2d(self.in_channels, inner_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(inner_channels * block.expansion)
            )
        else:
            projection = None

        layers = []
        layers.append(block(self.in_channels, inner_channels, stride, projection))
        self.in_channels = inner_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, inner_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for ResNet
        ResNet의 순전파
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResNetModel:
    """
    Factory class to create different ResNet variants
    다양한 ResNet 변형을 생성하는 팩토리 클래스
    """
    layers_count = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }
    
    def __init__(self, layers_count=50, num_classes=1000, zero_init_residual=True):
        """
        Initialize a ResNet model of specified depth
        지정된 깊이의 ResNet 모델 초기화
        
        Args:
            layers_count: Model depth (18, 34, 50, 101, or 152) | 모델 깊이 (18, 34, 50, 101, 또는 152)
            num_classes: Number of output classes | 출력 클래스 수
            zero_init_residual: Whether to initialize residual branch with zeros | 잔차 분기를 0으로 초기화할지 여부
        """
        if layers_count not in self.layers_count:
            raise ValueError(f"ResNet-{layers_count} is not supported. Available models: {list(self.layers_count.keys())}")
        
        # Select block type based on model depth | 모델 깊이에 따라 블록 유형 선택
        block = BasicBlock if layers_count <= 34 else BottleNeck
        
        # Create and store the model | 모델 생성 및 저장
        self.model = ResNet(block, self.layers_count[layers_count], num_classes, zero_init_residual)
    
    def get_model(self):
        """
        Return the initialized ResNet model
        초기화된 ResNet 모델 반환
        """
        return self.model