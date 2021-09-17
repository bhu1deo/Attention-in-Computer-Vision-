# Attention-in-Computer-Vision-

I implemented the following architectures all from scratch: 

a.) ResNet : https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf

b.) SENet : https://arxiv.org/abs/1709.01507

c.) NLNet : https://arxiv.org/abs/1711.07971

d.) Self Attention in Visual Models : https://arxiv.org/abs/1906.05909


The Dataset : https://drive.google.com/drive/folders/12pzB91LhyzfWPlPpheu9iWi2tlKRyOm_?usp=sharing
Flowers dataset, both the jpg format and the HDF5 formats have been provided. 

Results : On the Flowers Resized Dataset (224x224) I got the following accuracies(test dataset):

1.) Custom ResNet : 66%
2.) Pytorch ResNet : 68%
3.) SENet : 72%
4.) Single Block NLNet : 68%
5.) Single Block Self-Attention : 67%

The Last two architectures are not exact, and only single Blocks in the ResNet architecture are altered to spped up the training, hence the compromise. 

These codes are just a proof of the concept and do not represent actual efficient implementations. 
