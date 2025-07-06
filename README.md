# EfficientAD Project

이 프로젝트는 EfficientAD 모델을 구현하고 학습하며 배포하는 것을 목표로 합니다.

## 설치

프로젝트를 실행하기 위해 다음 라이브러리들을 설치해야 합니다.

1.  `requirements.txt` 파일의 라이브러리 설치:

    ```bash
pip install -r requirements.txt
    ```

2.  추가 라이브러리 설치:

    ```bash
pip install kornia open_clip_torch openvino
    ```

## Git 설정

학습 데이터, 결과물 및 가상 환경 파일이 Git 저장소에 커밋되지 않도록 `.gitignore` 파일이 설정되어 있습니다.

```
# Ignore data and results directories
efficient_ad_project/data/
efficient_ad_project/results/

# Ignore virtual environment
vvenv/
```

## 데이터 생성

`efficient_ad_project/tools/generate_synthetic_data.py` 스크립트를 사용하여 합성 데이터를 생성할 수 있습니다.

### 사용법

데이터를 생성하려면 `--output_dir` 인자를 사용하여 생성된 데이터를 저장할 디렉토리를 지정해야 합니다.

```bash
python efficient_ad_project/tools/generate_synthetic_data.py --output_dir .\efficient_ad_project\data\pcb\
```

**예시:**

현재 프로젝트의 `efficient_ad_project/data/synthetic` 디렉토리에 데이터를 생성하려면 다음 명령어를 사용합니다:

```bash
python efficient_ad_project/tools/generate_synthetic_data.py --output_dir F:/Source/EfficientAD/efficient_ad_project/data/synthetic
```

**선택적 인자:**

*   `--num_train`: 학습 데이터셋의 이미지 수 (기본값: 스크립트 내부 확인)
*   `--num_test_normal`: 정상 테스트 데이터셋의 이미지 수 (기본값: 스크립트 내부 확인)
*   `--num_test_anomalous`: 비정상 테스트 데이터셋의 이미지 수 (기본값: 스크립트 내부 확인)
*   `--image_size`: 생성될 이미지의 크기 (예: `256 256`)

**예시 (모든 인자 포함):**

```bash
python efficient_ad_project/tools/generate_synthetic_data.py --output_dir F:/Source/EfficientAD/efficient_ad_project/data/synthetic --num_train 1000 --num_test_normal 200 --num_test_anomalous 50 --image_size 256 256
```

## Imagenette 데이터셋 사용 (선택 사항)

이 프로젝트는 기본적으로 `pcb` 데이터셋을 사용하도록 설정되어 있습니다. `Imagenette`와 같은 다른 데이터셋을 사용하려면 다음 단계를 따르세요.

1.  **데이터셋 다운로드 및 구조화:**
    `Imagenette` 데이터셋을 다운로드한 후, `anomalib.data.Folder`가 인식할 수 있는 디렉토리 구조로 재구성해야 합니다. `anomalib.data.Folder`는 `root` 경로 아래에 `train/good`, `test/anomaly`, `test/good`와 같은 하위 디렉토리 구조를 기대합니다.

    ```
    <your_imagenette_root_path>/
    ├── train/
    │   └── good/
    │       ├── class1_image1.jpg
    │       └── ...
    ├── test/
    │   ├── good/
    │   │   ├── class1_image_test1.jpg
    │   │   └── ...
    │   └── anomaly/  # 필요시 비정상 이미지 추가
    │       ├── anomalous_image1.jpg
    │       └── ...
    ```

    `Imagenette`는 이상 감지용 데이터셋이 아니므로, `test/anomaly` 디렉토리에 비정상 이미지를 직접 생성하거나 다른 소스에서 가져와야 합니다.

2.  **`pcb_config.yaml` 업데이트:**
    `efficient_ad_project/configs/pcb_config.yaml` 파일에서 `data.path`를 `Imagenette` 데이터셋의 루트 경로로 변경합니다.

    ```yaml
data:
  source: real
  path: <your_imagenette_root_path> # Imagenette 데이터셋의 실제 경로로 변경
  category: imagenette # 또는 원하는 카테고리 이름
  image_size: [256, 256]
  train_batch_size: 1
  eval_batch_size: 32
    ```

## 모델 학습

`efficient_ad_project/tools/train.py` 스크립트를 사용하여 모델을 학습할 수 있습니다.

### 사용법

학습을 시작하려면 `--config` 인자를 사용하여 설정 파일 경로를 지정해야 합니다.

```bash
python efficient_ad_project/tools/train.py --config .\efficient_ad_project\configs\pcb_config.yaml
```

### 문제 해결 (Troubleshooting)

*   **`ModuleNotFoundError: No module named 'lightning'` 또는 `TypeError: Trainer.__init__() got an unexpected keyword argument 'task'` 오류:**
    `anomalib`와 `lightning` (또는 `pytorch-lightning`) 버전 간의 호환성 문제로 인해 발생할 수 있습니다. 이 문제를 해결하기 위해 `efficient_ad_project/tools/train.py` 파일에서 `Engine` 초기화 시 `task` 및 `Trainer`에 직접 전달되던 불필요한 인자들을 제거했습니다.

    **수정 전 (`efficient_ad_project/tools/train.py`):**
    ```python
    engine = Engine(
        task="classification", # or "segmentation"
        image_size=tuple(config["data"]["image_size"]),
        train_batch_size=config["data"]["train_batch_size"],
        eval_batch_size=config["data"]["eval_batch_size"],
        model=model,
        lr=config["model"]["lr"],
        max_epochs=config["trainer"]["max_epochs"],
        # Add other engine parameters from config as needed
    )
    ```

    **수정 후 (`efficient_ad_project/tools/train.py`):**
    ```python
    engine = Engine(
        # Add other engine parameters from config as needed
    )
    ```

    또한, `anomalib` 내부의 `lightning` import 경로 문제 (`ModuleNotFoundError: No module named 'lightning.pytorch.accelerators'`)를 해결하기 위해 `F:/Source/EfficientAD/venv/Lib/site-packages/anomalib/engine/accelerator/xpu.py` 파일의 import 문을 다음과 같이 수정했습니다.

    **수정 전 (`xpu.py`):**
    ```python
    from lightning.pytorch.accelerators import Accelerator, AcceleratorRegistry
    ```

    **수정 후 (`xpu.py`):**
    ```python
    from lightning.fabric.accelerators import Accelerator, AcceleratorRegistry
    ```

    이러한 수정은 `anomalib`와 `lightning` 라이브러리 간의 버전 불일치로 인한 문제를 해결하기 위한 것입니다. 만약 다른 버전의 라이브러리를 사용하거나 새로운 오류가 발생하면, 해당 라이브러리의 공식 문서를 참조하여 호환되는 버전을 확인하거나 추가적인 수정이 필요할 수 있습니다.

## 모델 테스트

학습된 모델을 사용하여 이미지를 테스트하려면 `efficient_ad_project/tools/inference.py` 스크립트를 사용합니다.

### 사용법

`inference.py` 스크립트는 이제 단일 이미지 또는 이미지 디렉토리를 처리할 수 있으며, 결과 이미지에 이물 위치와 추론 시간을 표시하고, 전체 평균 추론 시간을 보고합니다.

```bash
python efficient_ad_project/tools/inference.py --model_path [모델 경로] --input_path [이미지/디렉토리 경로] --output_path [결과 파일/디렉토리 경로]
```

**인자:**

*   `--model_path`: 학습된 모델 체크포인트(`.ckpt`) 파일의 경로
*   `--input_path`: 테스트할 이미지 파일 또는 이미지들이 포함된 디렉토리의 경로
*   `--output_path`: 결과 이미지를 저장할 파일 경로 (단일 이미지 입력 시) 또는 디렉토리 경로 (디렉토리 입력 시). 디렉토리 입력 시, 원본 디렉토리 구조를 유지하며 결과가 저장됩니다.

**예시:**

`bottle` 데이터셋으로 학습된 모델을 사용하여 테스트 데이터셋 전체를 추론하고, 결과를 `F:/Source/EfficientAD/results_Test` 디렉토리에 저장하는 예제입니다.

```bash
python F:/Source/EfficientAD/efficient_ad_project/tools/inference.py --model_path F:/Source/EfficientAD/results/EfficientAd/MVTecAD/bottle/v1/weights/lightning/model.ckpt --input_path F:/Source/EfficientAD/datasets/mvtec_anomaly_detection/bottle/test --output_path F:/Source/EfficientAD/results_Test
```

이 명령을 실행하면 `F:/Source/EfficientAD/inference_results` 디렉토리 내에 `broken_large`, `broken_small`, `contamination`, `good`과 같은 하위 디렉토리가 생성되고, 각 원본 이미지에 해당하는 결과 이미지가 해당 하위 디렉토리에 저장됩니다. 각 결과 이미지에는 이물 위치가 노란색 윤곽선으로 표시되며, 개별 추론 시간도 함께 표시됩니다. 모든 추론이 완료되면 전체 이미지에 대한 평균 추론 시간이 콘솔에 출력됩니다.

## ONNX 모델 변환 및 추론

EfficientAD 모델을 ONNX 형식으로 변환하고, 변환된 ONNX 모델을 사용하여 추론을 수행하는 방법을 설명합니다.

### ONNX 모델 변환

`efficient_ad_project/deployment/export_onnx.py` 스크립트를 사용하여 학습된 PyTorch 모델을 ONNX 형식으로 변환할 수 있습니다.

#### 사용법

```bash
python efficient_ad_project/deployment/export_onnx.py
```

이 스크립트는 `F:/Source/EfficientAD/results/EfficientAd/MVTecAD/bottle/v1/weights/lightning/model.ckpt` 경로의 모델을 `F:/Source/EfficientAD/results_Test/bottle_onnx_export/model.onnx` 경로로 변환합니다. 변환된 ONNX 모델은 `results_Test/bottle_onnx_export/` 디렉토리에 저장됩니다.

### ONNX 모델 추론

`efficient_ad_project/tools/inference_onnx.py` 스크립트를 사용하여 변환된 ONNX 모델로 추론을 수행할 수 있습니다.

#### 사용법

```bash
python efficient_ad_project/tools/inference_onnx.py --model_path [ONNX 모델 경로] --input_path [이미지/디렉토리 경로] --output_path [결과 파일/디렉토리 경로]
```

**인자:**

*   `--model_path`: ONNX 모델(`.onnx`) 파일의 경로
*   `--input_path`: 테스트할 이미지 파일 또는 이미지들이 포함된 디렉토리의 경로
*   `--output_path`: 결과 이미지를 저장할 파일 경로 (단일 이미지 입력 시) 또는 디렉토리 경로 (디렉토리 입력 시). 디렉토리 입력 시, 원본 디렉토리 구조를 유지하며 결과가 저장됩니다.

**예시:**

변환된 `bottle_onnx_export/model.onnx` 모델을 사용하여 `mvtec_anomaly_detection/bottle/test` 데이터셋 전체를 추론하고, 결과를 `F:/Source/EfficientAD/results_Test/bottle_test_onnx` 디렉토리에 저장하는 예제입니다.

```bash
python F:/Source/EfficientAD/efficient_ad_project/tools/inference_onnx.py --model_path F:/Source/EfficientAD/results_Test/bottle_onnx_export/model.onnx --input_path F:/Source/EfficientAD/datasets/mvtec_anomaly_detection/bottle/test --output_path F:/Source/EfficientAD/results_Test/bottle_test_onnx
```

이 명령을 실행하면 `F:/Source/EfficientAD/results_Test/bottle_test_onnx` 디렉토리 내에 `broken_large`, `broken_small`, `contamination`, `good`과 같은 하위 디렉토리가 생성되고, 각 원본 이미지에 해당하는 결과 이미지가 해당 하위 디렉토리에 저장됩니다. 각 결과 이미지에는 이물 위치가 노란색 윤곽선으로 표시되며, 개별 추론 시간도 함께 표시됩니다. 모든 추론이 완료되면 전체 이미지에 대한 평균 추론 시간이 콘솔에 출력됩니다.