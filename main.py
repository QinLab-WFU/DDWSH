import argparse
from train.hash_train import DDWSHTrainer
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="flickr", help="choice from [coco, flickr, nuswide, iapr]")
    parser.add_argument("--output-dim", type=int, default=16)
    parser.add_argument("--is-train", default=True)

    parser.add_argument("-backbone", type=str, default="clip")
    parser.add_argument("-preload", type=str, default="ViT-B-32.pt")
    args = parser.parse_args()

    DDWSHTrainer(args, 0)

if __name__ == "__main__":
    main()
