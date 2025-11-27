import os
import os.path as osp
import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict
import logging

logger = logging.getLogger('FL_face.dataset')

# ===============================================================
#  JSON Image Dataset (우리가 직접 만든 Dataset)
# ===============================================================
class JSONImageDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, "r") as f:
            self.paths = json.load(f)

        # Normalize path
        self.paths = [osp.normpath(p) for p in self.paths]

        # ID (폴더명) → int label 매핑
        self.id2label = {}
        self.labels = []
        self.ID_base = self.id2label

        for p in self.paths:
            cls_name = osp.basename(osp.dirname(p))
            if cls_name not in self.id2label:
                self.id2label[cls_name] = len(self.id2label)
            self.labels.append(self.id2label[cls_name])

        self.num_classes = len(self.id2label)

        # transforms
        self.transform = transform or transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        import PIL.Image as Image
        img_path = self.paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

# ===============================================================
#  Federated Dataset Manager (All_Client_Dataset)
#  - train_loaders
#  - test_loaders
#  - train_class_sizes
#  - train_dataset_sizes
# ===============================================================
class All_Client_Dataset(object):
    def __init__(self, root_dir, local_rank, args):
        """
        root_dir = cfg.rec (train.py에서 전달됨)
                   하지만 우리는 split 경로(JSON) 기준으로 새로 해석할 것
        """
        self.args = args
        self.local_rank = local_rank
        self.num_client = args.num_client
        self.batch_size = args.batch_size

        # JSON split root
        # 사용자 split 결과: ./data/split/client_000/train_list.json
        self.split_root = "./data/split"

        # transforms
        self.transform = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])

        # 생성
        self.creating_each_client()
        self.creating_infer_each_client()

    # -----------------------------------------------------------
    # Create train dataset per client
    # -----------------------------------------------------------
    def creating_each_client(self):
        self.train_loaders = []
        self.train_dataset_sizes = []
        self.train_class_sizes = []

        for cid in range(self.num_client):
            json_path = osp.join(self.split_root, f"client_{cid:03d}", "train_list.json")

            dataset = JSONImageDataset(json_path, transform=self.transform)
            loader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=2,
                                pin_memory=True)

            self.train_loaders.append(loader)
            self.train_dataset_sizes.append(len(dataset))
            self.train_class_sizes.append(dataset.num_classes)

        logger.info("----------- Train Dataset Info -----------")
        logger.info(f"Num Clients : {self.num_client}")
        logger.info(f"Train classes per client: {self.train_class_sizes[0]}")
        logger.info(f"Total IDs: {sum(self.train_class_sizes)}")
        logger.info("-----------------------------------------")

    # -----------------------------------------------------------
    # Create test dataset per client (same as train, no shuffle)
    # -----------------------------------------------------------
    def creating_infer_each_client(self):
        self.test_loaders = []

        for cid in range(self.num_client):
            json_path = osp.join(self.split_root, f"client_{cid:03d}", "train_list.json")

            dataset = JSONImageDataset(json_path, transform=self.test_transform)
            loader = DataLoader(dataset,
                                batch_size=256,
                                shuffle=False,
                                num_workers=2,
                                pin_memory=False)

            self.test_loaders.append(loader)

        logger.info("----------- Test Dataset Info -----------")
        logger.info("Test loaders created for each client")
        logger.info("----------------------------------------")
