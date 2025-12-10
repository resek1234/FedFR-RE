from easydict import EasyDict as edict

config = edict()
config.dataset = "ms1m_split"
config.embedding_size = 512
config.sample_rate = 1
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.lr = 0.05
config.step = [6,14]

config.rec = "./data/split"

# ===== LFW 평가용 경로 =====
# 너의 Windows 환경 기준 (수정 가능)
config.val_rec = "D:/FedFR-RE/data/lfw/lfw-deepfunneled"
config.val_pairs = "D:/FedFR-RE/data/lfw/pairs.txt"
config.val_targets = ["lfw"]

# local verification (쓰지 않음)
config.local_rec = ""

# federated 학습 관련 값
config.num_epoch = 16

def lr_step_func(epoch):
    return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
        [m for m in config.step if m - 1 <= epoch])
config.lr_func = lr_step_func

config.com_batch_size = 256
config.public_batch_size = 512
config.HN_threshold = 0.4
config.train_decay = 8
config.mu = 5
config.converter_layer = 1
