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

## 모델 학습

`efficient_ad_project/tools/train.py` 스크립트를 사용하여 모델을 학습할 수 있습니다.

### 사용법

학습을 시작하려면 `--config` 인자를 사용하여 설정 파일 경로를 지정해야 합니다.

```bash
python efficient_ad_project/tools/train.py --config .\efficient_ad_project\configs\pcb_config.yaml
```
