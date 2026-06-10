# 26-1IntermediateArtificialIntelligence

## 프로젝트 구조

```text
.
├── data/                       # 데이터셋 로딩 및 전처리
│   ├── load.py                 # 데이터셋별 전처리 결과 로더
│   ├── cifar100/
│   │   ├── classes.json        # CIFAR-100 클래스 정보
│   │   └── process.py          # CIFAR-100 다운로드 및 전처리
│   ├── fashion/
│   │   ├── classes.json        # Fashion-MNIST 클래스 정보
│   │   └── process.py          # Fashion-MNIST 다운로드 및 전처리
│   └── mnist/
│       ├── classes.json        # MNIST 클래스 정보
│       └── process.py          # MNIST 다운로드 및 전처리
├── docs/                       # 프로젝트 계획 및 워크플로 문서
│   ├── README.md
│   ├── implement_plan.md
│   ├── project_plan.pdf
│   └── workflow.md
├── models/                     # 이미지 분류 모델과 모델 설정
│   ├── __init__.py             # 모델 레지스트리 및 팩토리
│   ├── activations.py          # 활성화 함수 팩토리
│   ├── config.json             # 모델별 기본 하이퍼파라미터
│   ├── config.py               # 모델 설정 로드·검증·저장
│   ├── mlp.py                  # MLP 모델
│   ├── mlp_mixer.py            # MLP-Mixer 모델
│   ├── resnet.py               # ResNet 모델
│   └── simple_cnn.py           # 기본 CNN 모델
├── scripts/                    # 학습 및 평가 파이프라인
│   ├── train.py                # 학습 CLI 진입점
│   ├── trainer.py              # 학습·검증·테스트 루프
│   ├── eval.py                 # 저장된 실행 결과 평가
│   ├── evaluation.py           # 공통 평가 함수
│   ├── dataset.py              # Dataset 및 DataLoader 구성
│   ├── checkpoint.py           # 체크포인트 저장 및 복원
│   ├── diagnostics.py          # 가중치 진단 결과 생성
│   ├── history.py              # 학습 이력 저장
│   ├── logging_utils.py        # 로깅 설정
│   ├── metrics.py              # 분류 성능 지표
│   └── storage.py              # 실행 결과 디렉터리 관리
├── tests/                      # 데이터, 설정, 평가, TUI 테스트
├── tui.py                      # 학습·평가 실행용 curses TUI
├── tui_model_config.py         # TUI 모델 설정 및 과거 실행 관리
├── colab_execute.ipynb         # Google Colab 실행 노트북
├── requirements.txt            # Python 의존성
└── LICENSE
```

학습을 실행하면 결과는 기본적으로 다음 구조의 `storage/` 디렉터리에
생성됩니다.

```text
storage/{실행_이름}/
├── checkpoints/                # best/last 및 주기별 체크포인트
├── diagnostics/                # 가중치 시각화와 통계
├── config.json                 # 실행에 사용한 설정
├── history.csv
├── history.json
├── test_metrics.json
└── train.log
```
