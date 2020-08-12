# tiny-pix2pix

Redesigning the pix2pix model for a small image size dataset (like CIFAR-10) with fewer parameters and different PatchGAN architecture

# Important links
[Original paper](https://arxiv.org/abs/1611.07004) </br>
[Project page](https://phillipi.github.io/pix2pix/) </br>

# Model Architecture

## Generator

U-Net: The generator in pix2pix resembles an auto-encoder. The Skip Connections in the U-Net differentiate it from a standard Encoder-decoder architecture. The generator takes in the Image to be translated and compresses it into a low-dimensional, “Bottleneck”, vector representation. The generator then learns how to upsample this into the output image. The U-Net is similar to ResNets in the way that information from earlier layers are integrated into later layers. The U-Net skip connections are also interesting because they do not require any resizing, projections etc. since the spatial resolution of the layers being connected already match each other.


<p align="center">
  <img src="U-Net.png">
</p>


## Discriminator
PatchGAN: The discriminator used in pix2pix is another unique component to this design. The PatchGAN works by classifying individual (N x N) patches in the image as “real vs. fake”, opposed to classifying the entire image as “real vs. fake”. The authors reason that this enforces more constraints that encourage sharp high-frequency detail. Additionally, the PatchGAN has fewer parameters and runs faster than classifying the entire image.


<p align="center">
  <img src="PatchNet.png">
</p>


## tiny-pix2pix

<p align="center">
  <img src="tiny_pix2pix.png">
</p>

# Result
[Our paper](https://ieeexplore.ieee.org/document/9116887) </br>

We used this model to reconstruct the occluded portion of images to increase the performance of our classifier under this challenging condition.

<p align="center">
  <img src="cat_occlusion_reconstruction.png">
</p>

<p align="center">
  <img src="performance_model.png">
</p>
