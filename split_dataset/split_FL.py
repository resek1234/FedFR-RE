import os
import argparse
import random
import json
from glob import glob
from tqdm import tqdm
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_client", type=int, default=40)
    parser.add_argument("--num_ID", type=int, default=4000)
    return parser.parse_args()

def main():
    args = parse_args()

    train_root = os.path.join(args.data_dir, "train")
    output_root = args.output_dir

    os.makedirs(output_root, exist_ok=True)

    # 모든 ID 폴더 가져오기
    id_folders = sorted(glob(os.path.join(train_root, "*")))

    print(f"총 ID 개수: {len(id_folders)}")

    # num_ID 만큼만 사용
    if args.num_ID < len(id_folders):
        id_folders = id_folders[:args.num_ID]

    print(f"사용할 ID 개수: {len(id_folders)}")

    # 각 ID 별 이미지 목록 수집
    id_to_images = {}
    for id_path in tqdm(id_folders, desc="ID 스캔 중"):
        images = glob(os.path.join(id_path, "*.jpg"))
        images += glob(os.path.join(id_path, "*.png"))
        if len(images) > 0:
            id_to_images[id_path] = images

    # ID 리스트
    ids = list(id_to_images.keys())

    # 클라이언트 수만큼 분할
    random.shuffle(ids)
    client_splits = defaultdict(list)

    for i, id_path in enumerate(ids):
        client_splits[i % args.num_client].append(id_path)

    # 각 클라이언트 데이터 저장
    for c in range(args.num_client):
        client_dir = os.path.join(output_root, f"client_{c:03d}")
        os.makedirs(client_dir, exist_ok=True)

        split_list = []
        for id_path in client_splits[c]:
            for img_path in id_to_images[id_path]:
                split_list.append(img_path)

        # JSON 목록 저장
        with open(os.path.join(client_dir, "train_list.json"), "w") as f:
            json.dump(split_list, f, indent=2)

        print(f"클라이언트 {c:03d}: {len(split_list)}개 이미지 저장 완료")

    print("\n=== Split 완료! ===")
    print(f"저장 위치: {output_root}")

if __name__ == "__main__":
    main()
