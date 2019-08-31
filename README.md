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

<style>
<p align="center">
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_4 (InputLayer)           (None, 32, 32, 3)    0                                            
__________________________________________________________________________________________________
model_1 (Model)                (None, 32, 32, 3)    1941523     input_16[0][0]                   
__________________________________________________________________________________________________
model_2 (Model)                (None, 9, 9, 1)      1560897     input_16[0][0]                   
                                                                 model_10[1][0]                   
==================================================================================================
Total params: 3,502,420
Trainable params: 1,941,523
Non-trainable params: 1,560,897
__________________________________________________________________________________________________

</p>
</style>

# Result

It is trained on the Cifar10 dataset to reconstruct every input image(auto-encoding) and it has 0.006 MAE on the test set.
