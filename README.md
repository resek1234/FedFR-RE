본 레포지토리는 AAAI 2022에 게재된 논문  
**“FedFR: Joint Optimization Federated Framework for Generic and Personalized Face Recognition”**  
(https://arxiv.org/abs/2112.12496)을 기반으로 한 경량 재구현 버전입니다.

본 프로젝트는 **아나콘다(Conda) 기반 Python 환경에서 실행되도록 구성**되었으며,  
GPU가 없는 환경 또는 제한된 자원에서도 실행 가능한 단순화된 FedFR 학습 파이프라인을 제공합니다.  
연구·교육 목적의 재현을 목표로 하며, 원 논문의 대규모 실험 환경 전체를 수행하지는 않습니다.


---

## 🚀 프로젝트 주요 특징

- 아나콘다 기반 Python 환경에서 손쉽게 실행 가능
- FedFR 구조를 반영한 **Federated Learning 학습 루프 구현**
- **Global + Personalized 모델 구조 반영**
- **VGGFace2 데이터를 40개 클라이언트 / 4000 ID로 Federated Split**
- CPU 환경에서도 구동 가능하도록 경량화
- **LFW Generic Evaluation 지원**

## 📂 Federated Data Split (VGGFace2 기반)

본 프로젝트는 **VGGFace2 데이터셋을 기반으로 Federated Learning용 데이터를 생성**했습니다.  
아래 스크립트를 사용하여 **40명의 클라이언트 / 4000명의 ID**로 분할하였습니다.

> ⚠️ 단, **`--num_client`와 `--num_ID` 값은 사용자 환경에 맞게 조정하여 사용할 수 있습니다.**  
> 더 많은 클라이언트로 나누거나, 더 적은 ID만 사용할 수도 있습니다.
### 🔧 사용한 Split 명령어

```bash
python d:/FedFR-RE/split_dataset/split_FL.py \
    --data_dir "D:/FedFR-RE/data/vggface2" \
    --output_dir "D:/FedFR-RE/data/split" \
    --num_client 40 \
    --num_ID 4000
```
---

본 프로젝트는 아나콘다(Conda) 환경에서 다음과 같이 설정했습니다.
1) Conda 환경 생성
conda create -n fedfr python=3.8
2) 환경 활성화
conda activate fedfr


🏋️ 학습 실행 방법
python train.py --lr 0.003 --num_client 40 --local_epoch 1 --total_round 5
⚠️ num_client, local_epoch, total_round 등은 사용자 환경에 맞게 자유롭게 변경 가능합니다.
예: 라운드 수를 증가시키면 학습 수렴이 개선되며, 클라이언트 수 조절을 통해 FL 실험 규모를 손쉽게 변경할 수 있습니다.

📊 성능 평가 (LFW)

본 재구현 버전은 LFW 기반 Generic Embedding 성능 평가만 제공합니다.

1) 짧은 학습 (Short Training)

리소스 제약으로 인해 약 5 라운드 수준의 FL 학습만 수행했습니다.

2) IJB-C 평가 미지원

IJB-C Benchmark는:

복잡한 평가 프로토콜

대규모 메모리 필요
등의 이유로 포함하지 못했습니다.

3) DFC 모듈 미구현

논문의 고급 기능인 DFC(Decomposed Feature Calibration)는 복잡성 및 메모리 문제로 제외했습니다.

4) 데이터 규모 축소

원본 FedFR 실험 대비 4000 ID 정도의 축소된 데이터만 사용했습니다.

📁 데이터셋 다운로드 안내

본 저장소는 저작권 문제로 얼굴 이미지 데이터를 포함하지 않습니다.

VGGFace2: [https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/](https://academictorrents.com/details/535113b8395832f09121bc53ac85d7bc8ef6fa5b)

LFW: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset


