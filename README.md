# [Deep Distance Weighted Sampling Hashing for Cross-modal Retrieval](https://ieeexplore.ieee.org/document/11353914)
This paper is accepted for publication with IEEE TMM on January 16, 2026.

## Training

### Processing dataset
Before training, you need to download the oringal data from [coco](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)(include 2017 train,val and annotations), nuswide [Google drive](https://drive.google.com/file/d/11w3J98uL_KHWn9j22GeKWc5K_AYM5U3V/view?usp=drive_link), mirflickr25k [Baidu, 提取码:u9e1](https://pan.baidu.com/s/1upgnBNNVfBzMiIET9zPfZQ) or [Google drive](https://drive.google.com/file/d/18oGgziSwhRzKlAjbqNZfj-HuYzbxWYTh/view?usp=sharing) (include mirflickr25k and mirflickr25k_annotations_v080), then use the "data/make_XXX.py" to generate .mat file

### Download CLIP pretrained model
Pretrained model will be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). This code is based on the "ViT-B/32".

You should copy ViT-B-32.pt to this dir.

### Start

After the dataset has been prepared, we could run the follow command to train.
> python main.py --is-train --dataset coco --lr 0.001 --output-dim 64 --save-dir ./result/coco/64 --clip-path ./ViT-B-32.pt --batch-size 128


## Citation

@ARTICLE{11353914,
  author={Cheng, Kun and Qin, Qibing and Zhang, Wenfeng and Huang, Lei and Nie, Jie},
  journal={IEEE Transactions on Multimedia}, 
  title={Deep Distance Weighted Sampling Hashing for Cross-modal Retrieval}, 
  year={2026},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TMM.2026.3654452}}
