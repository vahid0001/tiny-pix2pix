# Tiny Pix2Pix

Reimplementation of the Pix2Pix model for a small image size dataset (like Cifar10) with fewer parameters and different PatchNet architecture

# Important links
[Google code[Tensoflow]](https://github.com/tensorflow/docs/blob/r2.0rc/site/en/r2/tutorials/generative/pix2pix.ipynb) </br>
[Original paper](https://arxiv.org/abs/1611.07004) </br>
[Project page](https://phillipi.github.io/pix2pix/) </br>

# Model Architecture

## Generator

U-Net: The Generator in pix2pix resembles an auto-encoder. The Skip Connections in the U-Net differentiate it from a standard Encoder-decoder architecture. The Generator takes in the Image to be translated and compresses it into a low-dimensional, “Bottleneck”, vector representation. The Generator then learns how to upsample this into the output image. The U-Net is similar to ResNets in the way that information from earlier layers are integrated into later layers. The U-Net skip connections are also interesting because they do not require any resizing, projections etc. since the spatial resolution of the layers being connected already match each other.


<p align="center">
  <img src="unet.png">
</p>


## Discriminator
Patch-Net: The PatchGAN discriminator used in pix2pix is another unique component to this design. The PatchGAN discriminator works by classifying individual (N x N) patches in the image as “real vs. fake”, opposed to classifying the entire image as “real vs. fake”. The authors reason that this enforces more constraints that encourage sharp high-frequency detail. Additionally, the PatchGAN has fewer parameters and runs faster than classifying the entire image.


<p align="center">
  <img src="patchnet.png">
</p>


## Pix2Pix

<p align="center">
  <img src="pix2pix.png">
</p>

# Result

The model is trained for 3 epochs on the Cifar10 dataset to reconstruct every input image (auto-encoding) and it can reconstruct images with the MAE of 0.0113 on the test set.
