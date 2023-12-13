import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms
import glob


class LIDCDataset(Dataset):
    def __init__(self, root_dir='../LIDC', augmentation=False):
        self.root_dir = root_dir
        self.paths = []
        self.paths = self.paths + self.get_paths(root_dir) + self.get_paths(os.path.join(root_dir, 'Clean'))
        self.augmentation = augmentation

    def get_paths(self, dir):
        image_path = os.path.join(dir, 'Image')
        mask_path = os.path.join(dir, 'Mask')
        studies = []
        for study in os.listdir(image_path):
            if os.path.isdir(os.path.join(image_path, study)):
                studies.append(study)
        paths = []
        for study in studies:
            files = []
            for file in os.listdir(os.path.join(image_path, study)):
                if file.endswith('.npy'):
                    files.append(file)
            files.sort()
            length_files = len(files)
            if length_files > 128:
                # center crop
                files = files[length_files // 2 - 64 : length_files // 2 + 64]
            elif length_files < 128:
                # pad to both sides
                padding_needed = 128 - length_files
                padding_left = padding_needed // 2
                padding_right = padding_needed - padding_left

                files = ["_pad_"] * padding_left + files + ["_pad_"] * padding_right

            for i in range(len(files)):
                if files[i] == "_pad_":
                    paths.append(("_pad_", "_pad_"))
                else:
                    paths.append((os.path.join(image_path, study, files[i]), os.path.join(mask_path, study, files[i])))
        return paths
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_path, mask_path = self.paths[index]
        if image_path == "_pad_":
            return {'data': torch.zeros(1, 128, 128)}
        img = np.load(image_path)

        if self.augmentation:
            random_n = torch.rand(1)
            if random_n[0] > 0.5:
                img = np.flip(img, 2)

        imageout = torch.from_numpy(img.copy()).float()
        imageout = imageout.unsqueeze(0)

        return {'data': imageout}

if __name__ == "__main__":
    ds = LIDCDataset(root_dir='/mnt/work/Code/lung-diffusion/data/lidc/data')
    print(len(ds))
    print(ds[0]['data'].shape)