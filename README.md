# DDWSH

If the paper is accepted, the complete code will be uploaded soon.

Training
Processing dataset
Before training, you need to download the oringal data from coco(include 2017 train,val and annotations), nuswide Google drive, mirflickr25k Baidu, 提取码:u9e1 or Google drive (include mirflickr25k and mirflickr25k_annotations_v080), then use the "data/make_XXX.py" to generate .mat file

Download CLIP pretrained model
Pretrained model will be found in the 30 lines of CLIP/clip/clip.py. This code is based on the "ViT-B/32".

You should copy ViT-B-32.pt to this dir.
