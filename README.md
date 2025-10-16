
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

```
python train.py --dataroot ./datasets/train/AB --name masked_pix2pix --model pix2pix --direction BtoA --preprocess resize --load_size 512 --crop_size 512 --gpu_ids 1 --batch_size 1 --n_epochs 200 --dataset_mode alignedmask --input_nc 2 --output_nc 1 --norm instance --gpu_ids 0

```

### Test

# Dataset

## Training Dataset

## Experimental Dataset

# Reference

This repository is a fork and extension of: [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
