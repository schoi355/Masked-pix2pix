
# Masked Pix2Pix

This repository implements a masked pix2pix algorithm, which is a modified version of the original [pix2pix model](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
The masked pix2pix framework enables selective image-to-image translation, where transformations are applied only to specified masked regions of the input image allowing more precise control over what parts of the image are modified.

This model is specifically developed to remove thermal reflections on passive infrared fiducial markers, where high emissivity contrast between materials (e.g., EPDM rubber vs. aluminum) can cause reflection artifacts in thermal imagery. By focusing translation within reflective regions, the masked pix2pix network reconstructs the true marker appearance while maintaining the original thermal context of the scene.

## Installation

For Conda users, you can create a new Conda environment by

```
git clone https://github.com/schoi355/Masked-pix2pix.git
cd Masked-pix2pix
conda env create -f environment.yml
```
and then activate the environment by
```
conda activate pytorch-img2img
```

## Usage
### Train
Refer to this document [TBD] to how to create the training data.
Training dataset is organized by
```
datasets/train/
├── AB/
├── mask/
├── original/
└── synthesized/
```
where `original` means the images without reflection and `synthesized` means the images with synthesized reflection on the marker. To combine the `original` and `synthesized` images for training, use this command:
```
python datasets/combine_A_and_B.py --fold_A datasets/train/original --fold_B datasets/train/synthesized --fold_AB datasets/train/AB
```

```
python train.py --dataroot ./datasets/train/AB --name masked_pix2pix --model pix2pix --direction BtoA --preprocess resize --load_size 512 --crop_size 512 --batch_size 1 --dataset_mode alignedmask --input_nc 2 --output_nc 1 --norm instance --epoch EPOCH_NUM
```

### Test

```
python test.py --dataroot $TEST_DATASET_FOLDER --name masked_pix2pix --model test --netG unet_256 --direction BtoA --preprocess resize --load_size 512 --crop_size 512 --no_dropout --dataset_mode single --norm instance --input_nc 1 --output_nc 1
```

# Dataset

## Training Dataset

## Experimental Dataset

# Reference

This repository is a fork and extension of: [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
