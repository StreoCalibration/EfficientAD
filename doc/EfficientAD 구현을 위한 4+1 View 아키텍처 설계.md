# **EfficientAD 구현을 위한 4+1 View 아키텍처 설계 (v2)**

## **1\. 개요 (Overview)**

본 문서는 "EfficientAD" 알고리즘을 실제 산업 현장에 적용하기 위한 시스템 아키텍처를 4+1 View 모델에 따라 기술한다. 이 설계의 목표는 고영테크놀러지의 AOI(자동 광학 검사) 장비 환경에서 PCB 및 반도체 웨이퍼의 미세한 결함을 밀리초 수준의 속도로 정확하게 탐지하는 시스템을 구축하는 것이다.

**v2 변경 사항**: 실제 데이터셋과 가상(Synthetic) 데이터셋을 동일한 인터페이스로 처리할 수 있는 **데이터 제공자(Dataset Provider)** 개념을 도입하여, 데이터 소스에 구애받지 않는 유연한 훈련 및 테스트 파이프라인을 구축한다.

* **Logical View**: 시스템의 주요 기능과 컴포넌트 구조를 정의한다.  
* **Process View**: 시스템의 런타임 동작, 즉 훈련 및 추론 프로세스의 흐름을 설명한다.  
* **Development View**: 소스 코드의 모듈화 및 개발 환경을 정의한다.  
* **Physical View**: 시스템이 배포될 하드웨어 및 네트워크 구성을 설명한다.  
* **Scenarios (+1 View)**: 주요 유스케이스를 통해 아키텍처의 타당성을 검증한다.

## **2\. 논리적 관점 (Logical View)**

시스템의 핵심 기능은 데이터 소스를 추상화하는 DatasetProvider와 EfficientAD 알고리즘의 구조를 따르는 모듈들로 구성된다.

| 모듈명 (Module Name) | 주요 기능 (Key Functions) | 관련 기술/라이브러리 |
| :---- | :---- | :---- |
| **데이터 제공자 (DatasetProvider)** | **\[신규\]** 데이터 소스(실제/가상)에 대한 추상 인터페이스 제공\<br\>- RealDatasetProvider: 실제 파일 시스템에서 이미지 로드\<br\>- SyntheticDatasetProvider: 가상 정상/비정상 이미지 동적 생성 | PyTorch DataLoader, OpenCV, NumPy |
| **특징 추출기 (PDN)** | \- 경량화된 특징 추출 네트워크 (Teacher/Student)\<br\>- ImageNet 사전 훈련된 모델로부터 지식 증류 | PyTorch, EfficientNet |
| **구조적 이상 탐지 모듈** | \- 학생-교사(Student-Teacher) 프레임워크\<br\>- 학생-교사 간 불일치를 통한 지역 이상 맵 생성 | PyTorch |
| **논리적 이상 탐지 모듈** | \- 오토인코더(Autoencoder) 기반 전역적 맥락 학습\<br\>- 예측 오류를 통한 전역 이상 맵 생성 | PyTorch |
| **손실 계산 모듈** | \- Hard Feature Loss, Pretraining Penalty 등 계산 | PyTorch |
| **추론 및 후처리 모듈** | \- 지역/전역 이상 맵 결합 및 최종 이상 점수 계산\<br\>- 이상 위치 시각화를 위한 히트맵 생성 | NumPy, OpenCV |

### **클래스 다이어그램 (Class Diagram)**

데이터 소스의 교체가 가능하도록 DatasetProvider 인터페이스를 도입하고, AnomalyDetector는 이 인터페이스에 의존하도록 변경한다.

classDiagram  
    direction LR

    class AnomalyDetector {  
        \-teacher: Teacher  
        \-student: Student  
        \-autoencoder: Autoencoder  
        \+train(dataset\_provider)  
        \+predict(image)  
    }

    class DatasetProvider {  
        \<\<Interface\>\>  
        \+get\_train\_loader()  
        \+get\_test\_loader()  
    }

    class RealDatasetProvider {  
        \+DatasetProvider  
        \-root\_path: string  
        \+get\_train\_loader()  
        \+get\_test\_loader()  
    }

    class SyntheticDatasetProvider {  
        \+DatasetProvider  
        \+generate\_normal\_image()  
        \+generate\_anomalous\_image()  
        \+get\_train\_loader()  
        \+get\_test\_loader()  
    }

    class FeatureExtractor {  
        \<\<Abstract\>\>  
    }  
    class Teacher {  
        \+FeatureExtractor  
    }  
    class Student {  
        \+FeatureExtractor  
    }  
    class Autoencoder {  
    }

    AnomalyDetector ..\> DatasetProvider : uses  
    AnomalyDetector \--\> Teacher  
    AnomalyDetector \--\> Student  
    AnomalyDetector \--\> Autoencoder

## **3\. 프로세스 관점 (Process View)**

프로세스는 이전과 동일하게 **훈련**과 **추론**으로 나뉘지만, 시작 단계에서 설정에 따라 다른 데이터 소스를 사용한다.

### **훈련 프로세스 (Training Process)**

1. **데이터 로딩**: **(변경)** 설정 파일(config.yaml)의 data.source 값에 따라 RealDatasetProvider 또는 SyntheticDatasetProvider가 선택된다. 선택된 제공자는 정상 훈련 이미지 배치와 외부 데이터(ImageNette) 배치를 생성/로드한다.  
2. **특징 추출**: 고정된 Teacher가 정상 이미지에서 특징 맵 T(I)를 추출한다.  
3. **학생/오토인코더 예측**: Student와 Autoencoder가 예측을 수행한다.  
4. **손실 계산**: LossCalculator가 손실을 계산한다.  
5. **가중치 업데이트**: Student와 Autoencoder의 가중치를 업데이트한다.

**GPU/CPU 가속기 설정:**
훈련은 `efficient_ad_project/tools/train.py` 파일 내에서 `anomalib.engine.Engine` 초기화 시 `accelerator` 파라미터를 통해 가속기를 지정합니다.
*   **CUDA (GPU) 사용:** `accelerator="cuda"`
*   **CPU 사용:** `accelerator="cpu"`

시스템에 CUDA 호환 GPU가 없거나 CUDA 관련 오류가 발생하는 경우, `train.py` 파일에서 `accelerator="cpu"`로 변경해야 합니다.

### **추론 프로세스 (Inference Process)**

추론 프로세스는 개별 이미지에 대해 작동하므로 데이터 제공자의 직접적인 영향은 없으나, 테스트 단계에서는 DatasetProvider를 통해 실제 또는 가상의 테스트셋을 불러와 모델 성능을 평가할 수 있다.

1. **이미지 입력**: AOI 장비로부터 검사할 이미지가 입력된다.  
2. **전방향 패스(Forward Pass)**: Teacher, Student, Autoencoder가 특징 맵을 생성한다.  
3. **이상 맵 계산**: 지역 및 전역 이상 맵을 계산한다.  
4. **결과 생성**: 최종 이상 맵과 점수를 생성한다.  
5. **결과 출력**: 이상 점수에 따라 정상/비정상 판정 및 결과를 출력한다.

## **4\. 개발 관점 (Development View)**

데이터 관련 로직을 src/data 디렉토리로 그룹화하고, 설정 파일에 데이터 소스를 선택하는 옵션을 추가한다.

### **디렉토리 구조 (Directory Structure)**

efficient\_ad\_project/  
├── configs/  
│   └── pcb\_config.yaml         \# (변경) data.source: real | synthetic 추가  
├── data/  
│   ├── pcb/  
│   │   ├── train/  
│   │   └── test/  
│   └── wafer/  
├── src/  
│   ├── \_\_init\_\_.py  
│   ├── data/                   \# (변경) 데이터 관련 모듈 그룹  
│   │   ├── \_\_init\_\_.py  
│   │   ├── provider.py         \# (변경) DatasetProvider 인터페이스 및 구현 클래스  
│   │   └── synthetic\_generator.py \# \[신규\] 가상 데이터 생성 로직  
│   ├── models/  
│   │   ├── \_\_init\_\_.py  
│   │   ├── efficient\_ad.py  
│   │   └── lightning\_model.py  
│   └── utils/  
│       └── loss.py  
├── tools/  
│   ├── train.py  
│   └── inference.py  
└── deployment/  
    └── export\_openvino.py

### **설정 파일 예시 (configs/pcb\_config.yaml)**

model:  
  name: efficient\_ad  
  model\_size: S  
  lr: 0.0001

data:  
  \# \[신규\] 'real' 또는 'synthetic' 중 선택하여 데이터 소스 지정  
  source: synthetic \# 또는 real  
    
  \# 'real' 소스일 경우 사용되는 설정  
  path: F:/Source/EfficientAD/datasets/mvtec_anomaly_detection  
  category: screw  
    
  \# 공통 설정  
  image\_size: \[256, 256\]  
  train\_batch\_size: 1  
  eval\_batch\_size: 32

trainer:  
  max\_epochs: 100

## **5\. 물리적 관점 (Physical View)**

물리적 관점은 변경되지 않는다. 훈련은 고성능 서버에서, 추론은 AOI 장비 내 임베디드 시스템에서 수행된다. 다만, 훈련 서버에서는 실제 데이터 없이도 가상 데이터 생성을 통해 모델 개발이 가능하다.

## **6\. 유스케이스 (+1 View)**

데이터 소스 추상화의 이점을 보여주는 새로운 유스케이스를 추가한다.

### **유스케이스 1: 새로운 PCB 모델에 대한 검사 모델 훈련 (실제 데이터)**

* **Actor**: 머신러닝 엔지니어  
* **Scenario**:  
  1. configs/pcb\_config.yaml 파일에서 data.source를 real로 설정한다.  
  2. (기존 시나리오와 동일하게 진행)

### **유스케이스 2: 생산 라인에서 실시간 불량 검출**

* (기존 시나리오와 변경 없음)

### **유스케이스 3: 가상 데이터로 모델 프로토타이핑 \[신규\]**

* **Actor**: 머신러닝 엔지니어  
* **Pre-condition**: 실제 PCB 데이터가 아직 준비되지 않았거나, 특정 유형의 결함을 시뮬레이션하고 싶다.  
* **Scenario**:  
  1. 엔지니어는 configs/pcb\_config.yaml 파일에서 data.source를 synthetic으로 설정한다.  
  2. 훈련 서버에서 python tools/train.py \--config configs/pcb\_config.yaml 명령을 실행한다.  
  3. **Logical View**: SyntheticDatasetProvider가 호출되어, 정상 이미지(예: 회색 사각형)와 비정상 이미지(예: 스크래치나 점이 추가된 사각형)를 동적으로 생성하여 훈련/테스트에 제공한다.  
  4. **Process View**: 전체 훈련 파이프라인이 실제 데이터 없이도 정상적으로 실행된다. 엔지니어는 이를 통해 모델 코드, 손실 함수, 훈련 로직의 버그를 사전에 디버깅할 수 있다.  
  5. 훈련이 완료되면, 가상 테스트셋에 대한 성능 지표가 출력되어 모델이 기본적인 이상 탐지 능력을 학습했는지 빠르게 확인할 수 있다.