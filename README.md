#DualCapsuleGAN

Implementation of [DuCaGAN: Unified Dual Capsule Generative Adversarial Network for Unsupervised Image-to-Image Translation] that learns a mapping from input images to output images. 

![]([https://github.com/linxi159/scGNGI/blob/master/figures/Figure_1_Final.jpg](https://github.com/linxi159/DualCapsuleGAN/blob/main/example_DuCaGAN.jpg)) 


## Setup

### Prerequisites
- Linux
- Python with numpy
- NVIDIA GPU + CUDA 8.0 + CuDNNv5.1
- TensorFlow 0.11

### Getting Started
- Clone this repo:
```bash
git clone git@github.com:yenchenlin/pix2pix-tensorflow.git
cd pix2pix-tensorflow
```
- Download the dataset (script borrowed from [torch code](https://github.com/phillipi/pix2pix/blob/master/datasets/download_dataset.sh)):
```bash
bash ./download_dataset.sh facades
```
- Train the model
```bash
python main.py --phase train
```
- Test the model:
```bash
python main.py --phase test
```

## Results
Here is the results generated from this implementation:

More results on other datasets coming soon!

**Note**: To avoid the fast convergence of D (discriminator) network, G (generator) network is updated twice for each D network update, which differs from original paper but same as [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow), which this project based on.

## Train
Code currently supports [CMP Facades](http://cmp.felk.cvut.cz/~tylecr1/facade/) dataset. To reproduce results presented above, it takes 200 epochs of training. Exact computing time depends on own hardware conditions.

## Test
Test the model on validation set of [CMP Facades](http://cmp.felk.cvut.cz/~tylecr1/facade/) dataset. It will generate synthesized images provided corresponding labels under directory `./test`.


## Acknowledgments
Code borrows heavily from [pix2pix](https://github.com/phillipi/pix2pix) and [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py). Thanks for their excellent work!

## License
MIT
