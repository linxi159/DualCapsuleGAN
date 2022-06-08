# DualCapsuleGAN

Implementation of [DuCaGAN: Unified Dual Capsule Generative Adversarial Network for Unsupervised Image-to-Image Translation] that learns a mapping from input images to output images. 
![](https://github.com/linxi159/DualCapsuleGAN/blob/main/example_DuCaGAN.jpg)

## Setup

### Prerequisites
- Linux
- Python with numpy
- NVIDIA GPU + CUDA 8.0 + CuDNNv5.1
- TensorFlow 0.11

### Getting Started
- Clone this repo:
```bash
git clone https://github.com/linxi159/DualCapsuleGAN.git
cd DualCapsuleGAN
```
- Download the dataset (script borrowed from [torch code](https://github.com/phillipi/pix2pix/blob/master/datasets/download_dataset.sh)):
```bash
bash ./download_dataset.sh facades
```
- Train the model
```bash
python main.py --
```
- Test the model:
```bash
python main.py --
```

## Results
Here is the results generated from this implementation:

More results on other datasets coming soon!

## Acknowledgments
Code borrows heavily from [pix2pix](https://github.com/phillipi/pix2pix) and [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py). Thanks for their excellent work!

## License
Apache License 2.0
