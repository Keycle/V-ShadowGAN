# V-ShadowGAN
==============<br>
by Ziyue Li
 <br>
  ***
## MTVSD dataset
----------------
  Our dataset is available for download at [Baidu Drive](https://pan.baidu.com/s/1G0CQRH8xWHwlIgaTQKaxAA)(8t6c).

## Training environment
  * Python 3.6
  * Pytorch 1.7.0
  * torchvision 0.8.1
  * numpy
<br>
## Train
    * Take 'shadow_train' and 'shadow_free' folders from 'all_data' folder respectively as training data and put them into MTVSD folder.<br>
    * Python train.py<br>
<br>
## Test
    * Take 'shadow_test' and 'free_test' folders from 'all_data' folder respectively as test data and put them into MTVSD folder.<br>
    * Python test.py<br>
<br>
## Acknowledgments
    Code is implemented based on [Mask-ShadowGAN: Learning to Remove Shadows from Unpaired Data](https://https://github.com/xw-hu/Mask-ShadowGAN). We would also like to thank the authors of [CycleGAN](https://arxiv.org/abs/1703.10593), Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros.
