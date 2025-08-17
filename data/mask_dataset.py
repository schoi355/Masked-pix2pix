from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
from torchvision import transforms
import os
import torch


class MaskDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = opt.dataroot
        self.dir_M = self.dir_A.replace("image", "mask")
        print(self.dir_M)
        
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        
        self.M_paths = [os.path.join(self.dir_M, os.path.basename(p)) for p in self.A_paths]

        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))
        self.transform_M = get_transform(opt, grayscale=True)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        M_path = self.M_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        M_img = Image.open(M_path).convert('L')

        A = self.transform(A_img)
        M = self.transform_M(M_img)
        A_with_mask = torch.cat((A, M), dim=0)

        return {'A': A_with_mask, 'A_paths': A_path, 'mask': M, 'mask_path': M_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
