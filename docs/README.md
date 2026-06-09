# Fashion-MNIST와 CIFAR-100 비교를 통한 이미지 분류 실험

MLP 기반 이미지 분류 모델이 단순한 데이터셋과 복잡한 데이터셋에서 어떤 한계를 보이는지 확인하고, CNN 계열 모델이 이를 얼마나 개선할 수 있는지 비교하기 위한 프로젝트입니다.

프로젝트 계획서는 [docs/project_plan.pdf](docs/project_plan.pdf)에 정리되어 있습니다.

## 프로젝트 목표

- Fashion-MNIST와 CIFAR-100 데이터셋을 비교하여 데이터 복잡도와 클래스 수가 모델 성능에 미치는 영향을 분석합니다.
- MLP 모델의 성능 한계를 확인하고, CNN 또는 ResNet 구조가 이미지 분류에서 제공하는 개선 효과를 실험합니다.
- Top-5 accuracy, class-per accuracy, 오분류 사례 분석을 통해 모델이 유사한 물체를 어떻게 구분하는지 평가합니다.

## 데이터셋

| 데이터셋 | 이미지 | 클래스 | 특징 |
| --- | --- | --- | --- |
| Fashion-MNIST | 28x28 grayscale | 10개 | 의류 이미지 분류 데이터셋 |
| CIFAR-100 | 32x32 RGB | 100개 | 클래스당 이미지 수가 적고 분류 난이도가 높음 |
| MNIST | 28x28 grayscale | 10개 | 예제 학습 코드에서 사용 |

데이터 로더는 `data.load.load()`로 사용할 수 있습니다.

```python
from data import load

(x_train, y_train), (x_test, y_test), classes = load("cifar100")
```

지원하는 이름은 `cifar100`, `fashion`, `fashionmnist`, `mnist`입니다.

## 저장소 구조

```text
.
├── data/
│   ├── load.py              # 전처리된 데이터셋 로더
│   ├── cifar100/process.py  # CIFAR-100 원본 파일을 pkl/json으로 변환
│   ├── fashion/process.py   # Fashion-MNIST 원본 파일을 pkl/json으로 변환
│   └── mnist/process.py     # MNIST 원본 파일을 pkl/json으로 변환
├── docs/
│   └── project_plan.pdf     # 프로젝트 계획서
├── models/
│   ├── config.json          # 모델별 기본 파라미터
│   ├── config.py            # 모델 설정 로더/저장기
│   └── {model_name}.py      # 모델 구현
├── train.py                 # 공통 학습 진입점
├── trainer.py               # 학습/검증/평가 루프
├── tui.py                   # 학습/평가/추론 스크립트 실행용 curses TUI
├── tui_model_config.py      # TUI 모델 설정 및 과거 실행 로더
└── requirements.txt
```

## 설치

이 프로젝트는 CUDA 13.2용 PyTorch wheel index를 사용합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`에는 다음 index 설정이 포함되어 있습니다.

```text
--index-url https://download.pytorch.org/whl/cu132
--extra-index-url https://pypi.org/simple
```

## 데이터 전처리

원본 데이터 파일이 각 데이터셋 디렉터리에 있는 경우 아래 명령으로 `train.pkl`, `test.pkl`, `classes.json`을 생성할 수 있습니다.

```bash
python data/fashion/process.py
python data/cifar100/process.py
python data/mnist/process.py
```

현재 저장소에는 전처리된 `pkl` 파일과 클래스 정보가 포함되어 있어 바로 `data.load.load()`로 읽을 수 있습니다.

## 학습 실행

모델 이름은 lowercase와 underscore만 사용하며 `models/{model_name}.py`에 대응합니다.

```bash
python train.py \
  --dataset fashion \
  --model_name mlp \
  --epochs 20 \
  --batch_size 256 \
  --learning_rate 1e-3
```

GPU를 사용하지 않으려면 다음처럼 실행합니다.

```bash
python train.py --dataset fashion --model_name mlp --gpu_id -1
```

모델 전용 파라미터는 `train.py`의 CLI가 아니라 `models/config.json`에서 관리합니다.

```json
{
  "mlp": {
    "flatten": true,
    "hidden_sizes": [512, 256, 128],
    "dropout_p": 0.3
  }
}
```

과거 실행 설정이나 별도 설정 파일을 사용하려면 다음 옵션을 지정합니다.

```bash
python train.py \
  --dataset fashion \
  --model_name mlp \
  --model_config_path storage/PREVIOUS_RUN/config.json
```

각 실행의 설정, history, checkpoint, weight 진단 결과는
`storage/{timestamp}_{dataset}_{model}/` 아래에 저장됩니다.

## TUI 실행

`tui.py`는 학습, 평가, 추론 스크립트를 curses 기반 화면에서 실행하기 위한 도구입니다. 스크립트 경로를 입력하면 argparse 옵션을 자동으로 읽고, 실행 중 로그와 CPU/GPU 사용량을 보여줍니다.

```bash
python tui.py
```

`model_config` 메뉴에서는 다음 작업을 지원합니다.

- `models/config.json`의 기본 설정 편집
- 이전 `storage/*/config.json` 설정 불러오기
- checkpoint에 저장된 설정 불러오기
- 모델 파라미터만 사용하거나 학습 설정 복제
- checkpoint 학습 재개 설정

`train`, `eval`, `print_inference` 메뉴에서는 실행할 스크립트와 인자를
선택합니다. GPU 정보는 `nvidia-smi`가 설치된 환경에서 표시됩니다.

## 실험 계획

프로젝트 계획서 기준 실험 흐름은 다음과 같습니다.

1. Fashion-MNIST와 CIFAR-100 데이터셋 EDA 및 비교 분석
2. 각 데이터셋에 대한 MLP와 CNN 모델 설계
3. Colab 또는 CUDA 사용 가능 환경 점검
4. Fashion-MNIST에서 MLP와 CNN 성능 비교
5. CIFAR-100에서 MLP와 CNN 성능 비교
6. Top-5 accuracy, class-per accuracy, 오분류 이미지로 결과 분석

## 참고 자료

- Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms: https://arxiv.org/abs/1708.07747
- The CIFAR-100 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- MLP-Mixer: An all-MLP Architecture for Vision: https://arxiv.org/abs/2105.01601
- Deep Residual Learning for Image Recognition: https://arxiv.org/abs/1512.03385
