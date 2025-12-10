import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import KFold

transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

def load_img(path):
    img = Image.open(path).convert("RGB")
    return transform(img)

def read_pairs(pairs_txt):
    pairs = []
    with open(pairs_txt, "r") as f:
        for line in f.readlines()[1:]:
            items = line.strip().split()
            pairs.append(items)
    return pairs

def get_paths(root, pairs):
    paths = []
    issame = []

    for p in pairs:
        if len(p) == 3:
            name, idx1, idx2 = p
            p1 = os.path.join(root, name, f"{name}_{int(idx1):04}.jpg")
            p2 = os.path.join(root, name, f"{name}_{int(idx2):04}.jpg")
            same = True
        else:
            n1, idx1, n2, idx2 = p
            p1 = os.path.join(root, n1, f"{n1}_{int(idx1):04}.jpg")
            p2 = os.path.join(root, n2, f"{n2}_{int(idx2):04}.jpg")
            same = False

        paths.append((p1, p2))
        issame.append(same)

    return paths, np.array(issame)

@torch.no_grad()
def extract_emb(model, paths):
    model.eval()
    embs = []

    for p1, p2 in paths:
        img1 = load_img(p1).unsqueeze(0).cuda()
        img2 = load_img(p2).unsqueeze(0).cuda()
        e1 = model(img1).cpu().numpy()
        e2 = model(img2).cpu().numpy()
        embs.append((e1, e2))

    return embs

def evaluate(embs, issame):
    scores = []
    for e1, e2 in embs:
        s = np.dot(e1, e2.T) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        scores.append(float(s))

    scores = np.array(scores)

    kfold = KFold(n_splits=10, shuffle=False)
    accs = []

    for train_idx, test_idx in kfold.split(scores):
        best_t = find_best_threshold(scores[train_idx], issame[train_idx])
        acc = calc_acc(best_t, scores[test_idx], issame[test_idx])
        accs.append(acc)

    return np.mean(accs), np.std(accs)

def find_best_threshold(scores, labels):
    thresholds = np.linspace(-1.0, 1.0, 2000)
    best_acc = 0
    best_t = 0
    for t in thresholds:
        pred = scores > t
        acc = np.mean(pred == labels)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t

def calc_acc(t, scores, labels):
    pred = scores > t
    return np.mean(pred == labels)
