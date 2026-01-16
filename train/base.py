import os
import re

import pandas as pd
from tqdm import tqdm
import torch

from utils import get_logger, get_summary_writer

from dataset.dataloader import dataloader
from torch.utils.data import DataLoader
import time
from utils.calc_utils import calc_map_k_matrix as calc_map_k

import scipy.io as scio


class TrainBase(object):

    def __init__(self, args):

        self.args = args
        os.makedirs(args.save_dir, exist_ok=True)
        self._init_writer()
        self.logger.info(self.args)
        self.rank = self.args.rank  # GPU ID

        self._init_dataset()
        self._init_model()

        self.global_step = 0
        # self.global_step_t = 0
        self.max_mapi2t = 0
        self.max_mapt2i = 0
        self.max_map = 0
        self.best_epoch_i = 0
        self.best_epoch_t = 0

    def _init_dataset(self):
        self.logger.info(f"init {self.args.dataset} dataset.")

        if self.args.dataset == 'flickr':
            data_dir = '/home/yuebai/Data/Dataset/CrossModel/MIRFLICKR-25K'
            self.args.nclass = 24
        elif self.args.dataset == 'coco':
            data_dir = '/home/yuebai/Data/Dataset/CrossModel/MS-COCO'
            self.args.nclass = 80
        elif self.args.dataset == 'nuswide':
            data_dir = '/home/yuebai/Data/Dataset/CrossModel/NUS-WIDE'
            self.args.nclass = 21
        elif self.args.dataset == 'iapr':
            data_dir = '/home/yuebai/Data/Dataset/CrossModel/IAPR'
            self.args.nclass = 291
        else:
            raise ValueError("Unknown dataset")
        self.index_file = os.path.join(data_dir, "index.mat")
        if 'nuswide' in self.args.dataset:
            self.caption_file = os.path.join(data_dir, "caption.txt")
        else:
            self.caption_file = os.path.join(data_dir, "caption.mat")
        self.label_file = os.path.join(data_dir, "label.mat")

        train_data, query_data, retrieval_data, collate_fn = dataloader(captionFile=self.caption_file,
                                                            indexFile=self.index_file,
                                                            labelFile=self.label_file,
                                                            maxWords=self.args.max_words,
                                                            imageResolution=self.args.resolution,
                                                            query_num=self.args.query_num,
                                                            train_num=self.args.train_num,
                                                            seed=self.args.seed)
        self.train_labels = train_data.get_all_label().to(self.rank)
        self.query_labels = query_data.get_all_label()
        self.retrieval_labels = retrieval_data.get_all_label()
        self.retrieval_num = len(self.retrieval_labels)
        self.logger.info(f"train shape: {self.train_labels.shape}")
        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")

        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True,
        )
        self.query_loader = DataLoader(
            dataset=query_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True,
        )
        self.retrieval_loader = DataLoader(
            dataset=retrieval_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True,
        )

        self.max_loader = DataLoader(
            dataset=train_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True,
        )

        self.n_samples = self.train_loader.dataset.__len__()

    def _init_model(self):
        self.model = None
        self.model_ddp = None

    def _init_writer(self):
        log_name = self.args.dataset + "-" + str(self.args.output_dim) + "-" + ("train.log" if self.args.is_train else "test.log")
        self.logger = get_logger(os.path.join(self.args.save_dir, log_name))
        self.writer = get_summary_writer(os.path.join(self.args.save_dir, "tensorboard"))

    def run(self):
        if self.args.is_train:
            self.train()
        else:
            for epoch in range(5):
                self.valid(epoch)

    def change_state(self, mode):

        if mode == "train":
            self.model.train()
            if hasattr(self, "criterion"):
                self.criterion.train()
        elif mode == "valid":
            self.model.eval()
            if hasattr(self, "criterion"):
                self.criterion.eval()

    @torch.no_grad()
    def get_code(self, data_loader, length: int):
        img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        encoder_time = 0
        for image, text, label, index in tqdm(data_loader):
            start_encoder_time = time.time()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            index = index.numpy()
            image_hash = self.model.encode_image(image)
            image_hash = torch.sign(image_hash)
            text_hash = self.model.encode_text(text)
            text_hash = torch.sign(text_hash)
            encoder_time += time.time() - start_encoder_time
            img_buffer[index, :] = image_hash.data
            text_buffer[index, :] = text_hash.data

        return img_buffer, text_buffer, encoder_time


    def save_model(self):
        save_dir = os.path.join(self.args.save_dir, "model")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pth"))
        self.logger.info("save mode to {}".format(os.path.join(self.args.save_dir, "model.pth")))

    def train_epoch(self, epoch):
        raise NotImplementedError("Function of 'train' doesn't implement.")

    def train(self):
        self.logger.info("Start train.")

        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            if hasattr(self.args, 'warmup') and epoch < self.args.warmup or not self.args.valid:
                continue
            self.valid(epoch)
            if self.args.save_csv:
                self.loss_records.append(self.epoch_loss.cpu().numpy())
                self.map_i2t_records.append(self.epoch_i2t.cpu().numpy())
                self.map_t2i_records.append(self.epoch_t2i.cpu().numpy())
        self.logger.info(
            f">>>>>>> FINISHED >>>>>> Best epoch, I-T: {self.best_epoch_i}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t}, mAP: {self.max_mapt2i}")
        if self.args.save_csv:
            results_df = pd.DataFrame({
                "Epoch": list(range(1, len(self.loss_records) + 1)),
                "Loss": self.loss_records,
                "MAP_i2t": self.map_i2t_records,
                "MAP_t2i": self.map_t2i_records
            })

            # 保存为CSV
            save_path = os.path.join(self.args.save_dir, "training_metrics.csv")
            results_df.to_csv(save_path, index=False)
            print(f"The indicator has been saved to {save_path}")

    def train_unseen(self):
        self.logger.info("Start train.")
        for i in range(len(self.train_loaders)):
            self.train_loader = self.train_loaders[i]
            self.query_loader = self.query_loaders[i]
            self.retrieval_loader = self.retrieval_loaders[i]
            for epoch in range(self.args.epochs):
                self.train_epoch(epoch)
                if hasattr(self.args, 'warmup') and epoch < self.args.warmup or not self.args.valid:
                    continue
                self.valid(epoch)
            self.logger.info(
                f">>>>>>> FINISHED >>>>>> Best epoch, I-T: {self.best_epoch_i}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t}, mAP: {self.max_mapt2i}")
            self.fold_map_i2t.append(self.max_mapi2t)
            self.max_mapi2t = 0
            self.fold_map_t2i.append(self.max_mapt2i)
            self.max_mapt2i = 0

    def valid_hook(self):
        pass

    @torch.no_grad()
    def valid(self, epoch=None):
        self.logger.info("Valid.")
        self.change_state(mode="valid")
        topk = 50 if self.args.unseen else None
        self.valid_hook()
        self.query_labels = self.query_loader.dataset.get_all_label()
        self.retrieval_labels = self.retrieval_loader.dataset.get_all_label()
        self.args.query_num = len(self.query_loader.dataset)
        self.retrieval_num = len(self.retrieval_loader.dataset)

        query_img, query_txt, q_encoder_time = self.get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt, r_encoder_time = self.get_code(self.retrieval_loader, self.retrieval_num)

        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, topk, self.rank)
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, topk, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, topk, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, topk, self.rank)

        self.epoch_i2t = mAPi2t
        self.epoch_t2i = mAPt2i

        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i = epoch
            if self.args.save_mat:
                self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t", add=mAPi2t)
            self.max_mapi2t = mAPi2t
        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t = epoch
            if self.args.save_mat:
                self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="t2i", add=mAPt2i)
            self.max_mapt2i = mAPt2i
        if self.max_map < mAPi2t + mAPt2i:
            self.max_map = self.max_mapt2i + self.max_mapi2t
            if self.args.save_model:
                self.save_model()
        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}\n"
            f">>>>>> MAX MAP(i->t): {self.best_epoch_i}:{self.max_mapi2t}, MAX MAP(t->i): {self.best_epoch_t}:{self.max_mapt2i}\n"
            f">>>>>> query_encoder_time: {q_encoder_time}, retrieval_encoder_time: {r_encoder_time}")

    @torch.no_grad()
    def test(self, mode_name="i2t"):
        if self.args.pretrained == "":
            raise RuntimeError("test step must load a model! please set the --pretrained argument.")
        self.change_state(mode="valid")
        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)
        query_img, query_txt, q_encoder_time = self.get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt, r_encoder_time = self.get_code(self.retrieval_loader, self.retrieval_num)

        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)

        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(f">>>>>> MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}")

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(
            os.path.join(save_dir, str(self.args.output_dim) + "-ours-" + self.args.dataset + "-" + mode_name + ".mat"),
            result_dict)
        self.logger.info(">>>>>> save all data!")

    def compute_loss(self):
        raise NotImplementedError("Function of 'compute_loss' doesn't implement.")

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t", add=None):
        save_dir = self.args.save_dir
        os.makedirs(save_dir, exist_ok=True)

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }

        prefix = f"{self.args.output_dim}-{self.args.method}-{self.args.dataset}-{mode_name}"

        if add is not None:
            best_existing_add = -float('inf')
            best_existing_path = None

            for fname in os.listdir(save_dir):
                if fname.startswith(prefix) and fname.endswith(".mat"):
                    match = re.match(rf"{re.escape(prefix)}-(\d+\.?\d*).mat", fname)
                    if match:
                        existing_add = float(match.group(1))
                        if existing_add > best_existing_add:
                            best_existing_add = existing_add
                            best_existing_path = os.path.join(save_dir, fname)

            if add > best_existing_add:
                if best_existing_path:
                    os.remove(best_existing_path)
                    self.logger.info(f"Deleted old file with lower performance: {best_existing_path}")

                new_path = os.path.join(save_dir, f"{prefix}-{add:.4f}.mat")
                scio.savemat(new_path, result_dict)
                self.logger.info(f">>>>>> Saved improved {mode_name} data with add={add:.4f}!")
            else:
                self.logger.info(
                    f"Skip saving {mode_name} data: add={add:.4f} <= existing best add={best_existing_add:.4f}")
        else:
            path = os.path.join(save_dir, f"{prefix}.mat")
            scio.savemat(path, result_dict)
            self.logger.info(f">>>>>> Saved {mode_name} data without performance check.")

