# 위의 코드에 이어서 실행
import openvino as ov

# OpenVINO 런타임 초기화 및 core 객체 생성
core = ov.Core()
import numpy as np

if "GPU" in core.available_devices:
    print("✅ GPU 감지 성공! GPU로 추론을 테스트합니다.")

    # OpenVINO는 GPU 추론 시 모델을 캐싱하여 다음 실행 속도를 높입니다.
    # 캐싱을 활성화하여 성능을 최적화할 수 있습니다.
    core.set_property("GPU", ov.properties.cache_mode("OPTIMAL"))

    # # 예시: 간단한 모델을 GPU에서 컴파일하는 코드
    # # 실제 사용 시에는 "your_model.xml" 부분에 모델 경로를 입력하세요.
    # try:
    #     # core.compile_model("your_model.xml", "GPU")
    #     print("GPU에서 모델 컴파일이 가능합니다.")
    # except Exception as e:
    #     print(f"모델 파일이 없어 컴파일은 건너뜁니다. GPU 설정은 유효합니다. {e}")

else:
    print("❌ 오류: GPU를 찾을 수 없습니다. 2단계 드라이버 설치를 다시 확인하세요.")