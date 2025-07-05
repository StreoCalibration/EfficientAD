# **EfficientAD: 밀리초 수준의 지연 시간에서 정확한 시각적 이상 탐지 \- 심층 분석 및 구현 가이드**

## **서론: 실시간, 고정밀 시각적 이상 탐지의 산업적 요구**

현대 산업 현장에서 자동화된 품질 관리는 생산 효율성과 직결되는 핵심 과제입니다. 특히, 실시간으로 결함을 감지해야 하는 비전 애플리케이션에서 이상 탐지(Anomaly Detection) 기술의 중요성은 날로 커지고 있습니다.1 수확기에 금속 물체가 유입되거나 제조 공정에서 오염이 발생하는 등, 결함을 늦게 발견할 경우 막대한 경제적 손실을 초래할 수 있기 때문입니다.1 이러한 배경에서 "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies" 연구는 산업계의 두 가지 핵심 요구사항, 즉 \*\*밀리초 수준의 빠른 처리 속도(latency)\*\*와

**최첨단(State-of-the-Art, SOTA) 수준의 높은 정확도**를 동시에 만족시키는 것을 목표로 제시되었습니다.1

### **비지도 이상 탐지(UAD) 패러다임**

대부분의 실제 산업 환경에서는 정상 제품의 데이터는 풍부하지만, 결함이 있는 비정상(anomalous) 데이터는 매우 드물고 그 종류 또한 예측하기 어렵습니다.1 이로 인해, 이상 탐지 모델은 주로

**비지도 학습(Unsupervised Learning)** 방식으로 개발됩니다. 이는 오직 정상, 즉 결함이 없는 이미지만을 사용하여 모델을 훈련하고, 훈련 데이터의 분포에서 벗어나는 패턴을 이상으로 간주하는 접근 방식입니다.1 EfficientAD 역시 이러한 패러다임을 따르며, 정상 이미지의 특징을 학습하여 테스트 시 정상 이미지와 다른 특징을 보이는 이미지를 이상으로 탐지합니다.

### **이상의 분류: 구조적 이상과 논리적 이상**

시각적 이상은 크게 두 가지 유형으로 나눌 수 있으며, EfficientAD는 이 두 유형을 모두 처리하기 위해 설계되었습니다.

* **구조적 이상(Structural Anomalies):** 이는 이미지의 국소적인 영역에서 발생하는 표면 수준의 결함을 의미합니다. 예를 들어, 제품의 긁힘(scratches), 눌림(dents), 오염(contaminations), 얼룩(stains) 등이 여기에 해당합니다.1 기존의 많은 이상 탐지 모델들이 주로 이 유형의 결함을 탐지하는 데 초점을 맞추었습니다.  
* **논리적 이상(Logical Anomalies):** 이는 개별 객체의 외형은 정상이지만, 객체 간의 관계나 맥락적 규칙을 위반하는 더 복잡하고 전역적인 수준의 결함입니다. 부품의 누락(missing objects), 잘못된 위치에 조립된 부품(misplaced components), 잘못된 순서나 배열, 혹은 나사의 길이가 다른 것과 같은 기하학적 제약 조건 위반 등이 포함됩니다.1

특히 논리적 이상의 개념은 MVTec LOCO AD 데이터셋의 등장을 통해 이상 탐지 연구의 새로운 도전 과제로 부상했습니다.6 기존 모델들은 이미지의 전역적인 맥락을 이해하는 데 한계가 있어 이러한 논리적 이상 탐지에 어려움을 겪었습니다. EfficientAD의 이중적 아키텍처, 즉 구조적 이상을 위한 학생-교사 모델과 논리적 이상을 위한 오토인코더의 결합은 이러한 연구 환경의 변화에 직접적으로 대응하기 위한 지능적인 설계의 결과물입니다. 이는 벤치마크 데이터셋의 발전이 어떻게 모델 아키텍처의 혁신을 이끌어내는지를 보여주는 명확한 사례라 할 수 있습니다.6

## **아키텍처 심층 분석: 학생-교사 모델과 오토인코더의 시너지**

EfficientAD의 핵심은 구조적 이상과 논리적 이상을 동시에, 그리고 효율적으로 탐지하기 위해 **패치 디스크립터 네트워크(Patch Descriptor Network, PDN)**, **학생-교사(Student-Teacher) 프레임워크**, 그리고 \*\*오토인코더(Autoencoder)\*\*를 유기적으로 결합한 데 있습니다.

### **패치 디스크립터 네트워크(PDN): 경량화된 특징 추출의 기반**

기존의 많은 이상 탐지 방법들이 WideResNet-101과 같이 깊고 무거운 사전 훈련된 네트워크를 특징 추출기로 사용했던 것과 달리, EfficientAD는 **PDN**이라는 매우 가볍고 효율적인 특징 추출기를 제안합니다.1

* **구조 및 훈련:** PDN은 단 4개의 컨볼루션 레이어로 구성된 "급격하게 깊이가 줄어든(drastically reduced depth)" 네트워크입니다.1 이 경량 네트워크는 ImageNet과 같은 대규모 데이터셋으로 사전 훈련된 더 깊은 네트워크(예: WideResNet-101)의 지식을 증류(distill)하는 방식으로 훈련됩니다.1 이 지식 증류 과정을 통해 PDN은 작고 빠른 구조를 유지하면서도 풍부한 표현력을 가진 특징을 추출할 수 있게 됩니다.  
* **정밀한 지역화:** PDN의 각 출력 뉴런은 33x33 픽셀의 명확하게 정의된 수용 필드(receptive field)를 가집니다.1 이는 이미지의 한 부분에서 발생한 이상이 멀리 떨어진 다른 정상 영역의 특징 벡터에 영향을 미치는 것을 방지하여, 매우 정밀한 이상 지역화(localization)를 가능하게 하는 핵심 요소입니다.

### **학생-교사 프레임워크: 구조적 이상 탐지**

EfficientAD는 지식 증류 기반의 학생-교사 프레임워크를 사용하여 구조적 이상을 탐지합니다.

* **핵심 메커니즘:** 사전 훈련되어 고정된(frozen) PDN이 '교사(Teacher)' 역할을 합니다. 그리고 교사와 동일한 구조를 가지지만 훈련 가능한 또 다른 PDN이 '학생(Student)' 역할을 합니다.1 학생 네트워크는 오직 정상 이미지에 대해서만 교사의 특징 출력을 정확하게 모방하도록 훈련됩니다.1  
* **이상 탐지 원리:** 훈련 과정에서 본 적 없는 이상(anomaly)이 포함된 이미지가 입력되면, 학생 네트워크는 교사의 출력을 제대로 예측하지 못하고 둘 사이에 큰 불일치(discrepancy)가 발생합니다. 이 특징 맵 간의 차이가 곧 이상 점수가 되어 '지역 이상 맵(local anomaly map)'을 형성합니다.1

### **논리적 이상 탐지를 위한 오토인코더**

논리적 이상을 탐지하기 위해 EfficientAD는 오토인코더를 매우 독창적인 방식으로 통합합니다.

* **목적:** 오토인코더는 객체의 누락, 잘못된 배치 등 이미지의 전역적인 맥락과 논리적 제약 조건을 학습하여 이를 위반하는 사례를 탐지하는 데 특화되어 있습니다.1  
* **지능적인 통합 방식:** 오토인코더는 정상 이미지에 대한 *교사 네트워크의 특징 출력*을 재구성하도록 훈련됩니다. 하지만 단순히 오토인코더의 재구성 결과와 교사의 출력 간의 차이를 이상 맵으로 사용하면, 정상 이미지에서도 발생하는 고유의 재구성 오류(예: 흐릿함) 때문에 오탐(false positive)이 발생할 수 있습니다.1 EfficientAD는 이 문제를 해결하기 위해,  
  **학생 네트워크가 교사의 출력뿐만 아니라 오토인코더의 출력까지 예측하도록** 훈련시킵니다. 이렇게 하면 학생은 정상 이미지에 대한 오토인코더의 체계적이고 예측 가능한 재구성 오류를 학습하게 됩니다. 따라서 테스트 시, 학생의 예측과 실제 오토인코더의 출력 간의 차이를 계산하면, 학생이 학습하지 못한 '비정상적인' 재구성 오류만 남게 되어 훨씬 깨끗한 '전역 이상 맵(global anomaly map)'을 얻을 수 있습니다.1

### **최종 이상 점수 계산**

최종적으로 지역 이상 맵(학생-교사 쌍에서 생성)과 전역 이상 맵(학생-오토인코더 쌍에서 생성)을 정규화한 후 결합(예: 평균)하여 종합적인 이상 맵을 만듭니다. 이미지 레벨의 이상 점수는 이 결합된 맵의 최댓값으로 결정됩니다.1

이러한 구조는 EfficientAD의 효율성이 단순히 개별 구성 요소의 경량화에서 비롯된 것이 아님을 보여줍니다. 오히려 PDN이라는 하나의 경량 아키텍처를 교사와 학생으로 재사용하고, 학생 네트워크에 교사 모방과 오토인코더 모방이라는 두 가지 작업을 동시에 부여하는 \*\*아키텍처 재사용(architectural reuse)\*\*과 \*\*공유 계산(shared computation)\*\*을 통해 계산 및 메모리 효율성을 극대화합니다. 이는 별개의 학생-교사 모델과 오토인코더 모델을 독립적으로 실행하는 것보다 훨씬 효율적이며, EfficientAD의 고속 성능을 유지하는 핵심적인 설계 철학입니다.

## **"손실 유도 비대칭성" 패러다임: 새로운 훈련 전략**

EfficientAD의 또 다른 핵심 혁신은 추론 시의 계산 비용을 전혀 증가시키지 않으면서 성능을 극대화하는 독창적인 손실 함수 설계에 있습니다. 이는 '손실 유도 비대칭성(Loss-induced asymmetry)'이라 불리는 개념으로 구현됩니다.1

### **대칭적 학생-교사 모델의 한계**

일반적으로 학생과 교사가 동일한 아키텍처를 가질 경우, 훈련 데이터가 많아지면 학생 네트워크의 성능이 너무 강력해져 정상 데이터의 분포를 넘어 비정상 데이터에 대해서까지 교사를 모방하는 과잉 일반화(over-generalization) 문제가 발생할 수 있습니다. 이 경우, 이상이 발생해도 학생-교사 간의 차이가 줄어들어 탐지 성능이 저하됩니다.1 기존 연구들은 이를 해결하기 위해 학생과 교사의 아키텍처를 다르게 설계하는(architectural asymmetry) 방식을 사용했지만, 이는 모델 설계를 제약하는 단점이 있습니다. EfficientAD는 아키텍처는 동일하게 유지하되, 손실 함수를 통해 비대칭적 효과를 유도합니다.

### **Hard Feature Loss (Lhard​): 어려운 문제에 집중하기**

* **개념:** 이 손실 함수는 학생이 교사를 모방하기 가장 어려워하는 특징, 즉 학생-교사 간의 차이가 가장 큰 특징에만 집중하여 학습하도록 유도합니다. 전체 특징 맵에 대해 평균적인 손실을 계산하는 대신, 가장 유익한 학습 신호에만 집중하는 방식입니다.1  
* **구현:** 훈련 이미지 I에 대해 교사 T와 학생 S의 출력 특징 맵 $T(I)$와 $S(I)$를 얻습니다. 각 위치 $(c, w, h)$에서의 제곱 오차 Dc,w,h​=(T(I)c,w,h​−S(I)c,w,h​)2를 계산합니다. 이후, 모든 D 값들 중에서 사전에 정의된 백분위수(예: 99.9%)에 해당하는 값 $d\_{hard}$를 찾고, 이 임계값을 초과하는 오차 값들만의 평균을 내어 최종 손실 $L\_{hard}$로 사용합니다.1Lhard​=mean({Dc,w,h​∣Dc,w,h​≥dhard​})

### **사전 훈련 페널티(Pretraining Penalty): 분포 외 데이터 망각하기**

* **개념:** 학생 네트워크가 훈련에 사용된 '정상' 데이터의 좁은 분포를 벗어나는 이미지에 대해서는 교사를 모방하지 못하도록 방해하는 페널티 항입니다.1  
* **구현:** 각 훈련 스텝마다, 정상 훈련 이미지와는 별개로 ImageNet과 같은 대규모 외부 데이터셋에서 무작위 이미지 P를 샘플링합니다. 이 이미지를 학생 네트워크에 통과시킨 후, 그 출력 특징 맵의 L2 norm 제곱 값을 계산하여 전체 손실에 더합니다. 이는 학생이 정상 분포 외의 이미지에 대해 큰 활성화 값을 생성하는 것을 억제하여, 오직 정상 데이터에만 특화되도록 만듭니다.1 전체 학생-교사 손실 $L\_{ST}$는 다음과 같이 표현됩니다.LST​=Lhard​+CWH1​c∑​∥S(P)c​∥F2​

  여기서 F는 프로베니우스 노름(Frobenius norm)을 의미합니다.

### **Ablation Study: 각 구성 요소의 기여도 정량화**

EfficientAD 논문에서 수행된 제거 연구(Ablation Study)는 이러한 새로운 기법들이 모델 성능에 얼마나 기여하는지를 명확하게 보여줍니다. 이 결과는 모델의 성능이 단순히 복잡한 아키텍처가 아닌, 지능적인 훈련 전략에서 비롯됨을 입증합니다.

| 구성 요소 | 기준 모델 대비 성능 향상 (Image-level AUROC) | 설명 |
| :---- | :---- | :---- |
| **기본 모델 (S-T only)** | \- | 학생-교사 프레임워크만 사용한 기준 성능 |
| **\+ Hard Feature Loss** | **\+ 1.0%** | 어려운 특징에 집중하는 손실 함수 추가 시 성능 향상 8 |
| **\+ Pretraining Penalty** | **\+ 0.4%** | 외부 데이터셋을 이용한 페널티 추가 시 성능 향상 8 |
| **\+ Autoencoder** | 논리적 이상 탐지 성능 대폭 향상 | 논리적 이상 탐지를 위한 오토인코더 모듈 추가 효과 1 |
| **\+ Quantile Normalization** | **\+ 0.7%** | 가우시안 정규화 대신 분위수 기반 정규화 사용 시 성능 향상 8 |

**Table 1: EfficientAD 제거 연구(Ablation Study) 결과 요약 (MVTec AD 데이터셋 기준)**

이 표에서 볼 수 있듯이, 추론 비용에 영향을 주지 않는 Hard Feature Loss만으로도 AUROC가 1.0%p나 향상되는 등, 손실 함수 설계의 혁신이 모델 성능에 결정적인 영향을 미쳤음을 알 수 있습니다. 이는 개발자에게 모델 아키텍처 구현만큼이나 손실 함수의 정확한 구현이 중요함을 시사합니다.

## **성능 분석 및 벤치마킹**

EfficientAD는 주요 산업용 이상 탐지 벤치마크 데이터셋에서 정확도와 속도 모두 새로운 기준을 제시했습니다.

### **평가 지표**

* **Image-level AUROC (Area Under the Receiver Operating Characteristic Curve):** 모델이 이미지 전체를 '정상' 또는 '비정상'으로 얼마나 잘 분류하는지를 나타내는 지표입니다. 1에 가까울수록 성능이 우수합니다.3  
* **Pixel-level AUPRO (Area Under the Per-Region Overlap Curve):** 모델이 이미지 내에서 이상의 위치를 얼마나 정확하게 픽셀 단위로 찾아내는지를 평가하는 지표입니다. 다양한 크기의 이상을 공정하게 평가하기 위해 AUROC보다 선호되기도 합니다.3

### **주요 벤치마크 성능 비교**

EfficientAD는 PatchCore, PaDiM, FastFlow 등 기존의 강력한 모델들과 비교하여 MVTec AD, VisA, MVTec LOCO AD 데이터셋에서 뛰어난 성능을 보였습니다. 특히 속도 면에서 압도적인 우위를 점하면서도 최상위권의 정확도를 유지하는 점이 주목할 만합니다.

| 모델 | 데이터셋 | Image AUROC (%) | Pixel AUPRO (%) | 처리량 (FPS) |
| :---- | :---- | :---- | :---- | :---- |
| **EfficientAD-S** | MVTec AD | 98.7 | 93.1 | **614** |
| **EfficientAD-M** | MVTec AD | 99.1 | \- | 269 |
| PatchCore | MVTec AD | 99.6 | 98.2 | 5.88 |
| PaDiM | MVTec AD | 95.3 | 97.5 | 4.4 |
| FastFlow | MVTec AD | 99.4 | 98.5 | 21.8 |
| **EfficientAD-S** | VisA | 97.5 | 93.1 | **614** |
| **EfficientAD-M** | VisA | 98.1 | 94.0 | 269 |
| PatchCore | VisA | \- | 98.1 | \- |
| **EfficientAD-S** | MVTec LOCO AD | 90.0 (Avg) | 78.4 (sPRO) | **614** |
| **EfficientAD-M** | MVTec LOCO AD | 90.7 (Avg) | 79.8 (sPRO) | 269 |
| PatchCore | MVTec LOCO AD | 80.3 (Avg) | 39.7 (sPRO) | \- |

**Table 2: EfficientAD와 주요 모델의 벤치마크 성능 비교** (데이터 출처: 3)

### **S 모델과 M 모델의 트레이드오프**

EfficientAD는 두 가지 버전으로 제공되어 사용자가 애플리케이션의 요구사항에 맞게 선택할 수 있습니다.

* **EfficientAD-S (Small):** 최대 속도에 최적화된 모델입니다. 초당 614개의 이미지를 처리하는 놀라운 속도로, 지연 시간에 극도로 민감한 실시간 시스템에 적합합니다.3  
* **EfficientAD-M (Medium):** 교사와 학생 네트워크의 커널 수를 두 배로 늘리고 일부 레이어를 추가하여 정확도를 높인 모델입니다. 초당 269개 이미지 처리 속도로 S 모델보다는 느리지만, 여전히 매우 빠르며 더 높은 정확도를 제공합니다.1

### **논문과 실제 구현 간의 속도 차이**

한 가지 주목해야 할 중요한 점은, 논문에서 보고된 밀리초 단위의 초고속 처리량과 anomalib과 같은 공개 라이브러리에서 사용자들이 경험하는 실제 속도 간에 상당한 차이가 있다는 것입니다.18 논문의 벤치마크는 최소한의 오버헤드를 가진 최적화된 추론 스크립트에서 측정된 반면,

anomalib은 PyTorch Lightning 프레임워크 기반으로 데이터 로딩, 전처리, 후처리, 로깅 등 다양한 부가 기능으로 인한 오버헤드가 발생합니다. 이로 인해 anomalib 환경에서의 추론 속도는 논문 수치보다 현저히 느리게 측정될 수 있습니다. 따라서 논문 수준의 최고 속도를 달성하기 위해서는, anomalib으로 훈련을 마친 모델을 OpenVINO와 같은 최적화된 추론 엔진용 포맷으로 내보내고(export), 프레임워크 오버헤드가 없는 경량 추론 스크립트를 사용하는 것이 필수적입니다. 이는 이론과 실제 배포 사이의 간극을 이해하는 데 중요한 시사점을 제공합니다.

## **anomalib을 이용한 실전 구현 가이드**

anomalib은 Intel에서 개발한 이상 탐지 딥러닝 라이브러리로, EfficientAD를 포함한 다양한 SOTA 알고리즘을 제공하여 연구 및 개발을 용이하게 합니다.20 이 섹션에서는

anomalib을 사용하여 EfficientAD를 구현하는 구체적인 방법을 단계별로 안내합니다.

### **환경 설정 및 의존성**

1. **anomalib 설치:** pip을 통해 간단하게 설치할 수 있습니다. 전체 기능을 사용하려면 full 옵션을 추가합니다.20  
   Bash  
   pip install anomalib\[full\]

2. **의존성 확인:** torch, torchvision, opencv-python, pyyaml 등 필수 라이브러리가 함께 설치됩니다.21  
3. **ImageNette 데이터셋 다운로드:** EfficientAD의 '사전 훈련 페널티' 손실 항을 계산하기 위해서는 외부 이미지 데이터셋이 필요합니다. anomalib은 기본적으로 ImageNet의 작은 서브셋인 ImageNette를 사용하며, 훈련 시 지정된 경로에 자동으로 다운로드합니다. config.yaml 파일에서 이 데이터셋의 경로를 올바르게 지정해야 합니다.11

### **실전 구현: 훈련 및 추론 명령어**

anomalib을 사용하여 EfficientAD 모델을 훈련하고 추론하는 구체적인 명령어는 다음과 같습니다.

#### **1. 훈련 (Training)**

훈련은 `efficient_ad_project/tools/train.py` 스크립트와 설정 파일(`configs/bottle_config.yaml`)을 사용하여 수행됩니다.

**명령어:**
```bash
python F:\Source\EfficientAD\efficient_ad_project\tools\train.py --config F:\Source\EfficientAD\efficient_ad_project\configs\bottle_config.yaml
```

**설명:**
*   `--config`: 훈련에 사용할 설정 파일의 경로를 지정합니다. 이 파일은 데이터셋 경로, 모델 크기, 학습률 등 훈련 관련 모든 파라미터를 정의합니다.
*   **데이터셋 경로 설정**: `configs/bottle_config.yaml` 파일 내 `data.path`는 훈련 데이터셋의 루트 경로를 지정합니다. 예를 들어, `F:/Source/EfficientAD/datasets/mvtec_anomaly_detection`으로 설정되어야 합니다.

**GPU/CPU 가속기 설정:**
`efficient_ad_project/tools/train.py` 파일 내에서 `anomalib.engine.Engine` 초기화 시 `accelerator` 파라미터를 통해 훈련에 사용할 가속기를 지정할 수 있습니다.
*   **CUDA (GPU) 사용:** `accelerator="cuda"`
*   **CPU 사용:** `accelerator="cpu"`

시스템에 CUDA 호환 GPU가 없거나 CUDA 관련 오류가 발생하는 경우, `train.py` 파일에서 `accelerator="cpu"`로 변경해야 합니다.

#### **2. 추론 (Inference)**

훈련된 모델을 사용하여 이미지에 대한 추론을 수행하려면 `efficient_ad_project/tools/inference.py` 스크립트를 사용합니다.

**명령어:**
```bash
python F:\Source\EfficientAD\efficient_ad_project\tools\inference.py --model_path F:\Source\EfficientAD\results\EfficientAd\MVTecAD\bottle\v1\weights\lightning\model.ckpt --image_path F:\Source\EfficientAD\datasets\mvtec_anomaly_detection\bottle\test\broken_large\000.png --output_path F:\Source\EfficientAD\results\EfficientAd\MVTecAD\bottle\v1\broken_large_000_result.png
```

**설명:**
*   `--model_path`: 훈련된 모델 체크포인트 파일(`.ckpt`)의 경로를 지정합니다.
*   `--image_path`: 추론을 수행할 이미지 파일의 경로를 지정합니다.
*   `--output_path`: 추론 결과(이상 맵, 점수 등이 시각화된 이미지)를 저장할 경로를 지정합니다.

### **anomalib 구현 코드 레벨 분석**


#### **torch\_model.py 분석**

이 파일은 EfficientAD의 순수 PyTorch 모델 아키텍처를 정의합니다 (EfficientAdModel 클래스).24

* **네트워크 정의:** EfficientAdModel 클래스 내부에서 teacher, student, autoencoder 네트워크가 정의됩니다. anomalib 구현에서는 논문의 PDN 대신 사전 훈련된 **EfficientNet**을 교사 네트워크의 백본으로 사용하는 실용적인 선택을 했습니다.11 학생 네트워크는 교사와 유사한 경량 구조를 가지며, 오토인코더는 컨볼루션 레이어로 구성됩니다.11 교사 네트워크는 훈련 중에 가중치가 고정됩니다.  
* **forward 메소드:** 이 메소드는 모델의 데이터 흐름을 정의합니다. 입력 batch와 '사전 훈련 페널티'를 위한 batch\_imagenet을 인자로 받습니다.24  
  1. 입력 이미지는 교사, 학생, 오토인코더를 각각 통과하여 특징 맵을 생성합니다.  
  2. compute\_student\_teacher\_distance를 통해 학생-교사 간의 거리(이상 맵)를 계산합니다.  
  3. 학생의 출력과 오토인코더의 출력을 비교하여 전역 이상 맵을 계산합니다.  
  4. compute\_losses 메소드를 호출하여 훈련 손실을 계산합니다.  
  5. compute\_maps를 통해 최종적으로 anomaly\_map\_st, anomaly\_map\_ae, 그리고 이 둘을 결합한 anomaly\_map\_combined를 포함하는 딕셔너리를 반환합니다.24

#### **lightning\_model.py 분석**

이 파일은 EfficientAdModel을 PyTorch Lightning 모듈로 감싸 훈련, 검증, 테스트 로직을 자동화합니다.11

* **training\_step 메소드:** 실제 손실 계산과 역전파가 이루어지는 곳입니다.  
  1. forward 패스를 통해 모델 출력을 얻습니다.  
  2. 논문의 **Hard Feature Loss**는 학생-교사 간 거리 맵(distance\_st)에서 높은 분위수(quantile) 값을 임계값으로 설정하고, 이보다 큰 값들만으로 손실을 계산하는 방식으로 구현됩니다.  
  3. 논문의 **Pretraining Penalty**는 batch\_imagenet을 학생 네트워크에 입력하여 얻은 특징의 L2 norm을 손실에 더하는 방식으로 구현됩니다.  
  4. 학생이 교사와 오토인코더의 출력을 모두 예측하도록 하는 학습은, loss\_st(학생-교사 손실)와 loss\_ae(학생-오토인코더 손실)를 각각 계산한 후 합산하여 최종 손실을 구성함으로써 이루어집니다.24

### **설정 및 훈련 (config.yaml)**

anomalib에서 모델 훈련은 주로 config.yaml 파일을 통해 제어됩니다. EfficientAD를 위한 주요 파라미터는 다음과 같습니다.

| 파라미터 경로 | 설명 | 예시 값 / 권장 사항 |
| :---- | :---- | :---- |
| model.name | 사용할 모델의 이름을 지정합니다. | efficient\_ad |
| model.model\_size | EfficientAD의 크기를 선택합니다 (속도 vs 정확도). | S 또는 M 24 |
| model.lr | 학습률(Learning Rate)을 설정합니다. | 0.0001 (기본값) 24 |
| model.weight\_decay | 옵티마이저의 가중치 감쇠(Weight Decay) 값입니다. | 1e-05 (기본값) 24 |
| data.path | 데이터셋이 위치한 경로입니다. | ./datasets/MVTec 27 |
| data.category | MVTec AD 데이터셋 내의 특정 카테고리를 지정합니다. | bottle 27 |
| data.image\_size | 모델에 입력될 이미지의 크기입니다. | \`\` 27 |
| data.train\_batch\_size | 훈련 시 배치 크기입니다. anomalib의 EfficientAD 구현에서는 **반드시 1로 설정**해야 합니다. | 1 28 |
| data.eval\_batch\_size | 검증 및 테스트 시 배치 크기입니다. | 32 |
| trainer.max\_epochs | 총 훈련 에포크 수입니다. | 100 |
| normalization.normalization\_method | 이상 맵 정규화 방법을 선택합니다. | min\_max 또는 cdf 30 |

**Table 3: anomalib의 config.yaml 내 EfficientAD 주요 하이퍼파라미터**

### **데이터 증강 전략**

anomalib은 albumentations 라이브러리를 통해 강력한 데이터 증강 기능을 지원하며, config.yaml 내 transform\_config 경로에 증강 설정 파일을 지정하여 사용할 수 있습니다.31

EfficientAD 논문 자체는 무작위 흑백 변환과 같은 최소한의 증강만을 사용했습니다.10 과도한 데이터 증강(예: 심한 색상 왜곡, 블러 처리)은 정상 이미지에 인위적인 이상 특징을 만들어낼 수 있으므로 주의해야 합니다. 특히 탐지하려는 실제 결함이 미세한 색상이나 질감 변화일 경우, 증강으로 인해 모델 성능이 저하될 수 있습니다.33 따라서 수평 뒤집기(Horizontal Flip)와 같은 구조를 해치지 않는 간단한 증강부터 시작하여 점진적으로 적용하는 것이 권장됩니다.

## **문제 해결, 알려진 이슈 및 최적화**

EfficientAD를 anomalib으로 구현할 때 커뮤니티에서 보고된 몇 가지 일반적인 문제와 해결책은 다음과 같습니다.

### **높은 메모리 사용량 (CUDA Out of Memory)**

* **문제:** 훈련 중, 특히 검증 에포크가 끝날 때마다 GPU 메모리 사용량이 계속 증가하여 결국 CUDA 메모리 부족(OOM) 오류가 발생하는 경우가 보고되었습니다. 이는 24GB VRAM을 가진 고사양 GPU에서도 발생할 수 있습니다.34  
* **원인:** 이 문제의 근본 원인은 anomalib의 후처리 및 메트릭 계산 로직에 있습니다. BinaryPrecisionRecallCurve와 같은 메트릭 객체들이 에포크가 진행되는 동안 예측값과 실제값 텐서를 리스트에 계속 누적하면서 메모리를 해제하지 않기 때문입니다.34  
* **해결책:** 커뮤니티에서 발견된 해결책은 메트릭 객체의 compute() 메소드를 호출한 직후, 캐시된 텐서를 비우기 위해 .reset() 메소드를 명시적으로 호출하는 것입니다. 장시간 훈련 시 이 수정은 필수적입니다.

### **예상보다 느린 추론 속도**

* **문제:** 앞서 4장에서 분석했듯이, anomalib의 기본 추론 파이프라인은 논문에서 주장하는 밀리초 수준의 속도보다 훨씬 느립니다.18  
* **해결책:** 배포 환경에서 최고의 성능을 얻기 위해서는 모델을 최적화된 형식으로 내보내는 과정이 필요합니다.

### **배포를 위한 OpenVINO로의 내보내기**

anomalib은 Intel 하드웨어에서 가속화된 추론을 위해 OpenVINO 포맷으로 모델을 내보내는 간소화된 경로를 제공합니다.20

1. anomalib의 표준 파이프라인을 사용하여 모델 훈련을 완료합니다.  
2. config.yaml 파일에 다음 설정을 추가합니다:  
   YAML  
   optimization:  
     export\_mode: openvino

3. 훈련이 끝나면 anomalib은 결과 폴더에 OpenVINO 추론에 필요한 model.xml과 model.bin 파일을 자동으로 생성합니다.  
4. 생성된 파일들을 OpenVINO 런타임과 함께 사용하면, PyTorch Lightning 프레임워크의 오버헤드가 제거된 경량화된 고성능 추론 애플리케이션을 구축할 수 있습니다.37

### **기타 알려진 이슈**

* **정사각형이 아닌 이미지:** 초기 버전에서는 오토인코더가 입력 크기의 첫 번째 차원(높이)만 사용하여 정사각형이 아닌 이미지(예: 512x256)에서 오류가 발생했습니다. 이 문제는 이후 수정되었습니다.25  
* **검증 데이터셋 누수:** 기본 설정이 훈련 데이터가 아닌 *테스트 데이터*의 일부를 검증용으로 분할하여 사용할 수 있다는 지적이 있었습니다. 이는 일종의 데이터 누수(data leakage)에 해당하므로, 신뢰할 수 있는 결과를 위해서는 반드시 **훈련 데이터셋에서 검증셋을 분할**하여 사용해야 합니다.39

## **고급 주제 및 하이퍼파라미터 튜닝**

사용자 정의 데이터셋에 대해 EfficientAD의 성능을 최적화하기 위해서는 전략적인 하이퍼파라미터 튜닝이 필요합니다.

### **전략적 튜닝 접근법**

1. **기준 모델 설정:** 먼저 기본 하이퍼파라미터로 모델을 훈련하여 성능 기준선(baseline)을 설정합니다. 이는 이후 튜닝의 효과를 측정하는 데 중요합니다.40  
2. **주요 파라미터 우선 튜닝:** 가장 영향력이 큰 model\_size, lr, weight\_decay와 같은 파라미터부터 튜닝을 시작합니다.

### **EfficientAD의 핵심 하이퍼파라미터**

* **model\_size ('S' vs. 'M'):** 속도와 정확도 간의 가장 중요한 트레이드오프입니다. 실시간 처리가 최우선이면 'S'를, 약간의 지연 시간을 감수하고 더 높은 정확도가 필요하면 'M'을 선택합니다.8  
* **lr (학습률):** 모델 수렴에 가장 결정적인 파라미터입니다. anomalib의 기본값인 1e-4에서 시작하여 데이터셋의 특성에 맞게 조정하는 것이 좋습니다.24 학습률 스케줄러(learning rate scheduler)를 함께 사용하면 더 안정적인 수렴을 기대할 수 있습니다.  
* **weight\_decay (가중치 감쇠):** 과적합을 방지하는 정규화 파라미터입니다. 기본값은 1e-5입니다.24  
* **image\_size:** 이미지 크기가 클수록 더 세밀한 특징을 포착할 수 있지만 계산 비용과 메모리 사용량이 증가합니다. 모델 아키텍처에 따라 입력 크기가 특정 숫자로 나누어 떨어져야 하는 제약이 있을 수 있습니다.41

### **자동화된 하이퍼파라미터 최적화(HPO)**

수동 튜닝이 비효율적일 경우, 자동화된 HPO 도구를 사용할 수 있습니다.

* **Grid Search / Randomized Search:** 정의된 범위 내에서 모든 조합(Grid) 또는 무작위 조합(Randomized)을 시도하는 기본적인 방법입니다.42  
* **Bayesian Optimization:** 이전 시도 결과를 바탕으로 다음 시도할 하이퍼파라미터 조합을 지능적으로 선택하는 고급 기법입니다. Optuna와 같은 라이브러리가 널리 사용됩니다.44  
* **anomalib과의 통합:** anomalib은 HPO CLI 명령과 sweep 설정 파일을 통해 Weights & Biases, Comet.ml과 같은 HPO 플랫폼과 쉽게 통합될 수 있습니다.20

## **EfficientAD의 유산: 후속 연구와 미래 방향**

EfficientAD는 발표 이후 산업 이상 탐지 분야의 중요한 기준으로 자리 잡았으며, 그 개념은 여러 후속 연구의 기반이 되었습니다.

### **PUAD: "그릴 수 없는" 논리적 이상 다루기**

* **문제 제기:** EfficientAD의 오토인코더는 여전히 공간적인 이상 맵 생성에 의존합니다. 하지만 "부품의 개수가 잘못된" 경우와 같이 일부 논리적 이상은 공간적으로 표현하기 어렵습니다. PUAD 연구는 이러한 이상을 **"그릴 수 없는(unpicturable)" 이상**으로 정의했습니다.5  
* **해결책:** PUAD는 EfficientAD를 특징 추출기로 직접 활용합니다. EfficientAD의 학생 또는 교사 네트워크에서 나온 특징 맵에 전역 평균 풀링(global average pooling)을 적용하여 단일 특징 벡터를 추출합니다. 이후 정상 데이터의 특징 벡터 분포를 다변량 가우시안으로 모델링하고, 테스트 시 입력 이미지의 특징 벡터가 이 정상 분포로부터 얼마나 떨어져 있는지를 \*\*마할라노비스 거리(Mahalanobis distance)\*\*로 계산하여 '그릴 수 없는' 이상의 점수로 사용합니다.5  
* **관계:** PUAD는 EfficientAD의 경쟁자가 아닌 **기능 확장**입니다. EfficientAD의 '그릴 수 있는' 이상 점수와 PUAD의 '그릴 수 없는' 이상 점수를 결합하여 MVTec LOCO AD 데이터셋에서 새로운 SOTA 성능을 달성했습니다.16 이는 EfficientAD가 학습한 특징의 견고함과 유용성을 입증하는 사례입니다.

### **CSAD: 구성 요소 수준의 추론을 위한 파운데이션 모델 통합**

* **문제 제기:** EfficientAD의 전역적 접근 방식은 이상 현상의 '논리'가 특정 부품(component)과 밀접하게 연관된 복잡한 장면에서는 한계를 가질 수 있습니다. 즉, 객체를 명시적으로 분할하거나 개수를 세지 않습니다.  
* **해결책:** CSAD는 새로운 패러다임을 제안합니다. Grounding DINO와 같은 강력한 파운데이션 모델을 비지도 방식으로 활용하여 **구성 요소 분할(component segmentation)을 위한 의사 레이블(pseudo-label)을 생성**합니다. 이 레이블로 경량 분할 네트워크를 학습시킨 후, 추론 시 이미지를 구성 요소들로 분할합니다. 이후 '패치 히스토그램(Patch Histogram)' 모듈이 구성 요소 유형의 히스토그램을 분석하여 위치나 수량 이상과 같은 논리적 이상을 효과적으로 탐지합니다.50  
* **패러다임의 전환:** EfficientAD가 오토인코더의 병목 현상을 통해 암묵적으로(implicitly) 전역 논리를 학습했다면, CSAD는 명시적인(explicit) 구성 요소 기반의 추론으로 나아갑니다. 이는 강력한 범용 파운데이션 모델의 등장이 이상 탐지 시스템에 장면 구성에 대한 강력한 의미론적 사전 지식(semantic prior)을 제공할 수 있게 되면서 가능해진 주요 연구 동향입니다.

### **현재의 한계와 미래 동향**

EfficientAD의 핵심은 여전히 재구성 또는 증류 오류에 기반하며, 이는 이상 현상의 강력한 대리 지표이지만 항상 완벽하지는 않습니다. CSAD와 같은 후속 연구에서 볼 수 있듯이, 향후 이상 탐지 기술은 대규모 비전-언어 모델(VLM)과 같은 파운데이션 모델을 통합하여, 이상 탐지 작업에 의미론적 이해와 명시적 추론 능력을 부여하는 방향으로 발전할 것입니다.50

## **배포를 위한 전략적 권장 사항**

### **모델 선택 체크리스트**

* **정확도 vs. 속도:** 5ms 미만의 지연 시간이 필수적인가? 그렇다면 **EfficientAD-S**로 시작하십시오. 정확도가 더 중요하고 5-10ms 정도의 지연 시간이 허용된다면 **EfficientAD-M**을 사용하십시오.8  
* **이상 유형:** 부품 누락/오배치와 같은 논리적 이상이 주요 관심사인가? 그렇다면 오토인코더가 활성화되었는지 확인하십시오. 만약 이상 현상이 부품 개수를 세는 등 매우 복잡한 논리를 포함한다면 **PUAD**나 **CSAD**와 같은 후속 연구 접근법을 고려하십시오.  
* **하드웨어:** Intel CPU/iGPU와 같은 엣지 디바이스에 배포할 경우, 최대 성능을 위해 **OpenVINO로 내보내기** 경로를 우선적으로 고려하십시오.37

### **프로젝트 시작 워크플로우**

1. **데이터셋 준비:** 대표적인 *정상* 이미지들로만 구성된 훈련 데이터셋을 수집합니다.  
2. **초기 훈련:** anomalib을 사용하여 기본 파라미터로 EfficientAD-S 모델을 훈련합니다.  
3. **기준 성능 평가:** 정상 및 비정상 샘플을 포함하는 소규모의 대표적인 테스트셋으로 초기 성능을 평가합니다.  
4. **문제 해결 및 튜닝:** 6장에서 설명한 가이드를 참조하여 성능 문제(속도, 메모리 등)를 해결합니다. 기준 성능이 부족할 경우 7장의 가이드에 따라 하이퍼파라미터를 튜닝합니다.  
5. **내보내기 및 배포:** 최종 모델을 OpenVINO와 같은 최적화된 형식으로 내보내어 실제 생산 환경에 통합합니다.

### **최종 평가**

EfficientAD는 최첨단 정확도와 실시간 성능이 상호 배타적이지 않음을 입증하며 산업 시각적 이상 탐지 분야에서 중추적인 역할을 했습니다. anomalib 라이브러리를 통해 높은 접근성을 제공하며, 그 개념적 프레임워크는 후속 고급 연구의 견고한 기반이 되었습니다. 따라서 EfficientAD는 오늘날에도 광범위한 산업 검사 작업에 있어 가장 먼저 고려해야 할 최상위 선택지 중 하나로 남아 있습니다.

#### **참고 자료**

1. EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies, 7월 4, 2025에 액세스, [https://openaccess.thecvf.com/content/WACV2024/papers/Batzner\_EfficientAD\_Accurate\_Visual\_Anomaly\_Detection\_at\_Millisecond-Level\_Latencies\_WACV\_2024\_paper.pdf](https://openaccess.thecvf.com/content/WACV2024/papers/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.pdf)  
2. EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies | Request PDF \- ResearchGate, 7월 4, 2025에 액세스, [https://www.researchgate.net/publication/369556965\_EfficientAD\_Accurate\_Visual\_Anomaly\_Detection\_at\_Millisecond-Level\_Latencies](https://www.researchgate.net/publication/369556965_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies)  
3. EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies, 7월 4, 2025에 액세스, [https://paperswithcode.com/paper/efficientad-accurate-visual-anomaly-detection](https://paperswithcode.com/paper/efficientad-accurate-visual-anomaly-detection)  
4. MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection, 7월 4, 2025에 액세스, [https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec\_ad.pdf](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_ad.pdf)  
5. efficientADAnomalyDetector \- Detect anomalies using EfficientAD network \- MATLAB, 7월 4, 2025에 액세스, [https://www.mathworks.com/help/vision/ref/efficientadanomalydetector.html](https://www.mathworks.com/help/vision/ref/efficientadanomalydetector.html)  
6. MVTec Loco AD \- Logical Constraints Anomaly Detection Dataset, 7월 4, 2025에 액세스, [https://www.mvtec.com/company/research/datasets/mvtec-loco](https://www.mvtec.com/company/research/datasets/mvtec-loco)  
7. MVTec LOCO AD Dataset \- Papers With Code, 7월 4, 2025에 액세스, [https://paperswithcode.com/dataset/mvtec-loco-ad](https://paperswithcode.com/dataset/mvtec-loco-ad)  
8. EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies | alphaXiv, 7월 4, 2025에 액세스, [https://www.alphaxiv.org/overview/2303.14535v3](https://www.alphaxiv.org/overview/2303.14535v3)  
9. rximg/EfficientAD: unofficial version of EfficientAD \- GitHub, 7월 4, 2025에 액세스, [https://github.com/rximg/EfficientAD](https://github.com/rximg/EfficientAD)  
10. distillation\_training.py \- rximg/EfficientAD \- GitHub, 7월 4, 2025에 액세스, [https://github.com/rximg/EfficientAD/blob/main/distillation\_training.py](https://github.com/rximg/EfficientAD/blob/main/distillation_training.py)  
11. anomalib/src/anomalib/models/image/efficient\_ad/lightning\_model.py at main \- GitHub, 7월 4, 2025에 액세스, [https://github.com/openvinotoolkit/anomalib/blob/main/src%2Fanomalib%2Fmodels%2Fimage%2Fefficient\_ad%2Flightning\_model.py](https://github.com/openvinotoolkit/anomalib/blob/main/src%2Fanomalib%2Fmodels%2Fimage%2Fefficient_ad%2Flightning_model.py)  
12. Classification: ROC and AUC | Machine Learning \- Google for Developers, 7월 4, 2025에 액세스, [https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)  
13. Fine-Tuning AnomalyCLIP: Class-Agnostic Zero-Shot Anomaly Detection \- LearnOpenCV, 7월 4, 2025에 액세스, [https://learnopencv.com/fine-tuning-anomalyclip-medical-anomaly-clip/](https://learnopencv.com/fine-tuning-anomalyclip-medical-anomaly-clip/)  
14. The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection, 7월 4, 2025에 액세스, [https://www.researchgate.net/publication/390247894\_The\_MVTec\_AD\_2\_Dataset\_Advanced\_Scenarios\_for\_Unsupervised\_Anomaly\_Detection](https://www.researchgate.net/publication/390247894_The_MVTec_AD_2_Dataset_Advanced_Scenarios_for_Unsupervised_Anomaly_Detection)  
15. MVTec AD Benchmark (Anomaly Detection) \- Papers With Code, 7월 4, 2025에 액세스, [https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad)  
16. MVTec LOCO AD Benchmark (Anomaly Detection) \- Papers With Code, 7월 4, 2025에 액세스, [https://paperswithcode.com/sota/anomaly-detection-on-mvtec-loco-ad](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-loco-ad)  
17. VisA Benchmark (Anomaly Detection) \- Papers With Code, 7월 4, 2025에 액세스, [https://paperswithcode.com/sota/anomaly-detection-on-visa](https://paperswithcode.com/sota/anomaly-detection-on-visa)  
18. EfficientAd is slower than other models in anomalib \#2147 \- GitHub, 7월 4, 2025에 액세스, [https://github.com/open-edge-platform/anomalib/discussions/2147](https://github.com/open-edge-platform/anomalib/discussions/2147)  
19. Slower inference/testing with EfficientAD than promised. · open-edge-platform anomalib · Discussion \#1183 \- GitHub, 7월 4, 2025에 액세스, [https://github.com/openvinotoolkit/anomalib/discussions/1183](https://github.com/openvinotoolkit/anomalib/discussions/1183)  
20. open-edge-platform/anomalib: An anomaly detection library comprising state-of-the-art algorithms and features such as experiment management, hyper-parameter optimization, and edge inference. \- GitHub, 7월 4, 2025에 액세스, [https://github.com/open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib)  
21. EfficientAD/requirements.txt at main \- GitHub, 7월 4, 2025에 액세스, [https://github.com/rximg/EfficientAD/blob/main/requirements.txt](https://github.com/rximg/EfficientAD/blob/main/requirements.txt)  
22. python tools/train.py \--model efficient\_ad \--config \*\*\*\*.yaml \#not train myself data · Issue \#1508 · open-edge-platform/anomalib \- GitHub, 7월 4, 2025에 액세스, [https://github.com/openvinotoolkit/anomalib/issues/1508](https://github.com/openvinotoolkit/anomalib/issues/1508)  
23. \[Bug\]: efficientad training its own dataset reports an error · Issue \#1177 \- GitHub, 7월 4, 2025에 액세스, [https://github.com/openvinotoolkit/anomalib/issues/1177](https://github.com/openvinotoolkit/anomalib/issues/1177)  
24. Efficient AD — Anomalib documentation \- Read the Docs, 7월 4, 2025에 액세스, [https://anomalib.readthedocs.io/en/v2.0.0/markdown/guides/reference/models/image/efficient\_ad.html](https://anomalib.readthedocs.io/en/v2.0.0/markdown/guides/reference/models/image/efficient_ad.html)  
25. \[Help\]\[EfficientAD\] How to change the input size of training processs? · Issue \#1352 · open-edge-platform/anomalib \- GitHub, 7월 4, 2025에 액세스, [https://github.com/openvinotoolkit/anomalib/issues/1352](https://github.com/openvinotoolkit/anomalib/issues/1352)  
26. Export the torch model of EfficientAD · openvinotoolkit anomalib · Discussion \#1337 \- GitHub, 7월 4, 2025에 액세스, [https://github.com/openvinotoolkit/anomalib/discussions/1337](https://github.com/openvinotoolkit/anomalib/discussions/1337)  
27. MVTec-AD : Anomaly Detection with Anomalib Library \- Kaggle, 7월 4, 2025에 액세스, [https://www.kaggle.com/code/scottsuk0306/mvtec-ad-anomaly-detection-with-anomalib-library](https://www.kaggle.com/code/scottsuk0306/mvtec-ad-anomaly-detection-with-anomalib-library)  
28. EfficientAd training via API slower than via CLI · Issue \#2218 · open-edge-platform/anomalib, 7월 4, 2025에 액세스, [https://github.com/openvinotoolkit/anomalib/issues/2218](https://github.com/openvinotoolkit/anomalib/issues/2218)  
29. Efficient AD — Anomalib 2022 documentation, 7월 4, 2025에 액세스, [https://anomalib.readthedocs.io/en/v1.1.0/markdown/guides/reference/models/image/efficient\_ad.html](https://anomalib.readthedocs.io/en/v1.1.0/markdown/guides/reference/models/image/efficient_ad.html)  
30. Why Predicted Heat maps trained using efficient\_ad models look strange. · open-edge-platform anomalib · Discussion \#1647 \- GitHub, 7월 4, 2025에 액세스, [https://github.com/openvinotoolkit/anomalib/discussions/1647](https://github.com/openvinotoolkit/anomalib/discussions/1647)  
31. Data Transforms \- Anomalib Documentation \- Read the Docs, 7월 4, 2025에 액세스, [https://anomalib.readthedocs.io/en/v2.0.0/markdown/guides/how\_to/data/transforms.html](https://anomalib.readthedocs.io/en/v2.0.0/markdown/guides/how_to/data/transforms.html)  
32. Parchcore Data Augmentation Example · open-edge-platform anomalib · Discussion \#737, 7월 4, 2025에 액세스, [https://github.com/openvinotoolkit/anomalib/discussions/737](https://github.com/openvinotoolkit/anomalib/discussions/737)  
33. Anomaly Detection with FiftyOne and Anomalib \- Voxel51, 7월 4, 2025에 액세스, [https://docs.voxel51.com/tutorials/anomaly\_detection.html](https://docs.voxel51.com/tutorials/anomaly_detection.html)  
34. \[Bug\]: EfficientAd \- CUDA out of memory. · Issue \#2531 · open-edge-platform/anomalib, 7월 4, 2025에 액세스, [https://github.com/openvinotoolkit/anomalib/issues/2531](https://github.com/openvinotoolkit/anomalib/issues/2531)  
35. \[Bug\]: torch.cuda.OutOfMemoryError: CUDA out of memory \- PatchCore · Issue \#2016 · open-edge-platform/anomalib · GitHub, 7월 4, 2025에 액세스, [https://github.com/openvinotoolkit/anomalib/issues/2016](https://github.com/openvinotoolkit/anomalib/issues/2016)  
36. \[Task\]: Why is the EfficientAD training so slow, run 200 epoch · Issue \#1164 \- GitHub, 7월 4, 2025에 액세스, [https://github.com/openvinotoolkit/anomalib/issues/1164](https://github.com/openvinotoolkit/anomalib/issues/1164)  
37. Export & Optimization \- Anomalib v0.3.7, 7월 4, 2025에 액세스, [https://anomalib.readthedocs.io/en/v0.3.7/tutorials/export.html](https://anomalib.readthedocs.io/en/v0.3.7/tutorials/export.html)  
38. Anomalib \- Release 2022 Intel OpenVINO, 7월 4, 2025에 액세스, [https://anomalib.readthedocs.io/\_/downloads/en/v1.1.2/pdf/](https://anomalib.readthedocs.io/_/downloads/en/v1.1.2/pdf/)  
39. \[Bug\]: anomaly map normalization in efficient ad looks wrong · Issue \#1370 · open-edge-platform/anomalib \- GitHub, 7월 4, 2025에 액세스, [https://github.com/openvinotoolkit/anomalib/issues/1370](https://github.com/openvinotoolkit/anomalib/issues/1370)  
40. Tips for Tuning Hyperparameters in Machine Learning Models \- MachineLearningMastery.com, 7월 4, 2025에 액세스, [https://machinelearningmastery.com/tips-for-tuning-hyperparameters-in-machine-learning-models/](https://machinelearningmastery.com/tips-for-tuning-hyperparameters-in-machine-learning-models/)  
41. Training EfficientDet on custom data with PyTorch-Lightning (using an EfficientNetv2 backbone) | by Chris Hughes | Data Science at Microsoft | Medium, 7월 4, 2025에 액세스, [https://medium.com/data-science-at-microsoft/training-efficientdet-on-custom-data-with-pytorch-lightning-using-an-efficientnetv2-backbone-1cdf3bd7921f](https://medium.com/data-science-at-microsoft/training-efficientdet-on-custom-data-with-pytorch-lightning-using-an-efficientnetv2-backbone-1cdf3bd7921f)  
42. A Comprehensive Guide to Hyperparameter Tuning in Machine Learning | by Aditi Babu, 7월 4, 2025에 액세스, [https://medium.com/@aditib259/a-comprehensive-guide-to-hyperparameter-tuning-in-machine-learning-dd9bb8072d02](https://medium.com/@aditib259/a-comprehensive-guide-to-hyperparameter-tuning-in-machine-learning-dd9bb8072d02)  
43. 3.2. Tuning the hyper-parameters of an estimator \- Scikit-learn, 7월 4, 2025에 액세스, [https://scikit-learn.org/stable/modules/grid\_search.html](https://scikit-learn.org/stable/modules/grid_search.html)  
44. Optimizing Performance: A Hands-On Guide to Hyperparameter Tuning \- Zerve, 7월 4, 2025에 액세스, [https://www.zerve.ai/blog/optimizing-performance-a-hands-on-guide-to-hyperparameter-tuning](https://www.zerve.ai/blog/optimizing-performance-a-hands-on-guide-to-hyperparameter-tuning)  
45. Hyperparameter Tuning in Python: a Complete Guide \- neptune.ai, 7월 4, 2025에 액세스, [https://neptune.ai/blog/hyperparameter-tuning-in-python-complete-guide](https://neptune.ai/blog/hyperparameter-tuning-in-python-complete-guide)  
46. Anomalib \- Comet Docs, 7월 4, 2025에 액세스, [https://www.comet.com/docs/v2/integrations/third-party-tools/anomalib/](https://www.comet.com/docs/v2/integrations/third-party-tools/anomalib/)  
47. Anomalib in 15 Minutes, 7월 4, 2025에 액세스, [https://anomalib.readthedocs.io/en/latest/markdown/get\_started/anomalib.html](https://anomalib.readthedocs.io/en/latest/markdown/get_started/anomalib.html)  
48. \[2402.15143\] PUAD: Frustratingly Simple Method for Robust Anomaly Detection \- arXiv, 7월 4, 2025에 액세스, [https://arxiv.org/abs/2402.15143](https://arxiv.org/abs/2402.15143)  
49. \[Literature Review\] PUAD: Frustratingly Simple Method for Robust Anomaly Detection, 7월 4, 2025에 액세스, [https://www.themoonlight.io/en/review/puad-frustratingly-simple-method-for-robust-anomaly-detection](https://www.themoonlight.io/en/review/puad-frustratingly-simple-method-for-robust-anomaly-detection)  
50. CSAD: Unsupervised Component Segmentation for Logical Anomaly Detection \- BMVA Archive, 7월 4, 2025에 액세스, [https://bmva-archive.org.uk/bmvc/2024/papers/Paper\_854/paper.pdf](https://bmva-archive.org.uk/bmvc/2024/papers/Paper_854/paper.pdf)  
51. CSAD: Unsupervised Component Segmentation for Logical Anomaly Detection \- arXiv, 7월 4, 2025에 액세스, [https://arxiv.org/abs/2408.15628](https://arxiv.org/abs/2408.15628)  
52. CSAD: Unsupervised Component Segmentation for Logical Anomaly Detection \- arXiv, 7월 4, 2025에 액세스, [https://arxiv.org/pdf/2408.15628?](https://arxiv.org/pdf/2408.15628)  
53. matlab-deep-learning/zero-shot-anomaly-classification-with-EfficientAD-and-LLM \- GitHub, 7월 4, 2025에 액세스, [https://github.com/matlab-deep-learning/zero-shot-anomaly-classification-with-EfficientAD-and-LLM](https://github.com/matlab-deep-learning/zero-shot-anomaly-classification-with-EfficientAD-and-LLM)