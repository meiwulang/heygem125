# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/landmark2face_wy/data/Facereala3dmm_dataset.py
# Compiled at: 2024-03-06 17:51:22
# Size of source mod 2**32: 9747 bytes
import os.path, random
from landmark2face_wy.data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from landmark2face_wy.data.image_folder import make_dataset
from PIL import Image, ImageEnhance
import numpy as np, cv2, torch

def get_idts(config_name):
    idts = list()
    with open(os.path.join("../config", config_name + ".txt")) as f:
        for line in f:
            line = line.strip()
            idts.append(line)

    return idts


def obtain_seq_index(index, num_frames):
    seq = list(range(index - 13, index + 13 + 1))
    seq = [min(max(item, 0), num_frames - 1) for item in seq]
    return seq


def get_audio_feature(img_path, idx, new_dict):
    id = img_path.split("/")[-3]
    features = new_dict[id]
    return np.array(features[:, idx[0]:idx[1]].transpose(1, 0))


config_name_dict = {
  '杨舒涵': 0, 'dwp': 1, 'aixia': 2, 'wild': 3, 'Recording': 4}

class Facereala3dmmDataset(BaseDataset):

    def __init__(self, opt, mode=None):
        BaseDataset.__init__(self, opt)
        img_size = opt.img_size
        idts = get_idts(opt.name.split("_")[0])
        print("---------load data list--------: ", idts)
        self.new_dict = {}
        if mode == "train":
            self.labels = []
            self.label_starts = []
            self.label_ends = []
            one_hot = None
            count = 0
            for idt_name in idts:
                for config_key in config_name_dict.keys():
                    if config_key in idt_name:
                        one_hot = np.zeros((27, len(config_name_dict)))
                        config_idx = config_name_dict[config_key]
                        one_hot[:, config_idx] = 1
                        break
                    root = os.path.join(opt.feature_path, idt_name)

                if opt.audio_feature == "3dmm":
                    training_data_path = os.path.join(root, "{}_{}.t7".format(img_size, mode))
                else:
                    training_data_path = os.path.join(root, "{}_{}_{}.t7".format(img_size, mode, opt.audio_feature))
                training_data = torch.load(training_data_path)
                img_paths = training_data["img_paths"]
                features_3dmm = np.asarray(training_data["features_3dmm"])
                index = [i[0].split("/")[-1] for i in img_paths]
                image_dir = "{}/{}_dlib_crop".format(root, img_size)
                self.label_starts.append(count)
                for img in range(len(index)):
                    img_path = os.path.join(image_dir, index[img])
                    idx_list = obtain_seq_index(img, features_3dmm.shape[0])
                    self.labels.append([img_path,
                     np.transpose(np.concatenate((features_3dmm[idx_list, 80:144], one_hot), axis=1), (1,
                                                                                  0))])
                    count = count + 1

                self.label_ends.append(count)

            self.label_starts = np.array(self.label_starts)
            self.label_ends = np.array(self.label_ends)
            self.transforms_image = transforms.Compose([transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.transforms_label = transforms.Compose([transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.shuffle()
        elif mode == "test":
            self.labels = []
            self.label_starts = []
            self.label_ends = []
            count = 0
            one_hot = None
            for idt_name in idts:
                for config_key in config_name_dict.keys():
                    if config_key in idt_name:
                        one_hot = np.zeros((27, len(config_name_dict)))
                        config_idx = config_name_dict[config_key]
                        one_hot[:, config_idx] = 1
                        break
                    root = os.path.join(opt.feature_path, idt_name)

                if opt.audio_feature == "3dmm":
                    training_data_path = os.path.join(root, "{}_{}.t7".format(img_size, mode))
                else:
                    training_data_path = os.path.join(root, "{}_{}_{}.t7".format(img_size, mode, opt.audio_feature))
                training_data = torch.load(training_data_path)
                img_paths = training_data["img_paths"]
                features_3dmm = np.asarray(training_data["features_3dmm"])
                index = [i[0].split("/")[-1] for i in img_paths]
                image_dir = "{}/{}_dlib_crop".format(root, img_size)
                self.label_starts.append(count)
                for img in range(len(index)):
                    img_path = os.path.join(image_dir, index[img])
                    idx_list = obtain_seq_index(img, features_3dmm.shape[0])
                    self.labels.append([img_path,
                     np.transpose(np.concatenate((features_3dmm[idx_list, 80:144], one_hot), axis=1), (1,
                                                                                  0))])
                    count = count + 1

                self.label_ends.append(count)

            self.label_starts = np.array(self.label_starts)
            self.label_ends = np.array(self.label_ends)
            self.transforms_image = transforms.Compose([transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.transforms_label = transforms.Compose([transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.shuffle()

    def shuffle(self):
        self.labels_index = list(range(len(self.labels)))
        random.shuffle(self.labels_index)

    def add_mouth_mask2(self, img):
        mask = np.ones_like(img)
        rect_area = [img.shape[1] // 2 - 60, np.random.randint(226, 246), 30, 226]
        mask_rect_area = mask[rect_area[0]:rect_area[1], rect_area[2]:rect_area[3]]
        x = np.tile(np.arange(rect_area[1] - rect_area[0])[:, np.newaxis], (1, rect_area[3] - rect_area[2]))
        x = np.flip(x, 0)
        y = np.tile(np.arange(rect_area[3] - rect_area[2])[:, np.newaxis], (1, rect_area[1] - rect_area[0])).transpose()
        zz1 = -y - x + 88 > 0
        zz2 = np.flip(zz1, 1)
        zz = zz1 + zz2 > 0
        mask[rect_area[0]:rect_area[1], rect_area[2]:rect_area[3]] = np.tile(zz[:, :, np.newaxis], (1,
                                                                                                    1,
                                                                                                    3)) * 1
        imgm = img * mask
        return imgm

    def __getitem__(self, index):
        idx = self.labels_index[index]
        (img_path, feature_3dmm) = self.labels[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        img = np.array((np.clip(img + np.random.randint((-20), 20, size=3, dtype="int8"), 0, 255)), dtype="uint8")
        cut_pad1 = np.random.randint(0, 10)
        cut_pad2 = np.random.randint(0, 10)
        img = img[cut_pad1:256 + cut_pad1, cut_pad2:256 + cut_pad2]
        mask_B = img.copy()
        mask_end = np.random.randint(216, 236)
        index = np.random.randint(75, 85)
        mask_B[mask_B.shape[1] // 2 - index:mask_end, 50:-50] = 0
        img = Image.fromarray(img)
        mask_B = Image.fromarray(mask_B)
        img = self.transforms_image(img)
        mask_B = self.transforms_image(mask_B)
        x = np.where((idx >= self.label_starts) * (idx < self.label_ends))[0]
        audio = torch.tensor(feature_3dmm)
        max_i = 0
        real_A_index = random.randint(self.label_starts[x], self.label_ends[x] - 1)
        while True:
            if not real_A_index == idx:
                max_i += 1
                real_A_index = random.randint(self.label_starts[x], self.label_ends[x] - 1)
                if max_i > 5:
                    break

        (imgA_path, _) = self.labels[real_A_index]
        imgA = np.array(Image.open(imgA_path).convert("RGB"))
        cut_pad1 = np.random.randint(0, 10)
        cut_pad2 = np.random.randint(0, 10)
        imgA = imgA[cut_pad1:256 + cut_pad1, cut_pad2:256 + cut_pad2]
        imgA = Image.fromarray(imgA)
        imgA = self.transforms_image(imgA)
        return {
          'A': imgA, 'A_label': audio, 'B': img, 'B_label': audio, 'mask_B': mask_B}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.labels)


if __name__ == "__main__":
    from options.train_options import TrainOptions
    opt = TrainOptions().parse()
    dataset = Facereala3dmmDataset(opt)
    dataset_size = len(dataset)
    print(dataset_size)
    for (i, data) in enumerate(dataset):
        print(data)

