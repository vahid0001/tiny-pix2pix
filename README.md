# Modified Pix2Pix

Reimplementation the Pix2Pix model for a small dataset with less parameters and different Patch-Net architecture

[Google code[Tensoflow]](https://github.com/tensorflow/docs/blob/r2.0rc/site/en/r2/tutorials/generative/pix2pix.ipynb) </br>
[Paper](https://arxiv.org/abs/1611.07004) </br>


U-Net:


Input shape:(32, 32, 3)    
              
Output shape:(32, 32, 3)

<p align="center">
  <img src="unet.png">
</p>


Patch-Net:


Input shape:(32, 32, 3), (32, 32, 3)    
              
Output shape:(9, 9, 1)

<p align="center">
  <img src="patchnet.png">
</p>


Pix2Pix:
<p align="center">
  <img src="pix2pix.png">
</p>

# Result

It is trained on the Cifar10 dataset to reconstruct every input image(auto-encoding) and it has 0.006 MAE on the test set.
