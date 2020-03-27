# Pre-defined-sparseCNN
This repository constains a Pytorch implementation of the paper, titled "Pre-defined Sparsity for Low-Complexity Convolutional Neural Networks" (https://ieeexplore.ieee.org/document/8988206) which is published as a journal paper in IEEE transactions on Computer, 2020. The work is extended version of our conference paper: "pSConv: A Pre-defined Sparse Kernel Based Convolution for Deep CNNs" (https://ieeexplore.ieee.org/document/8919683) published in Allerton Conference , 2019.

If you find this project useful to you, please cite our work:
Journal:
========
@ARTICLE{8988206, author={S. {Kundu} and M. {Nazemi} and M. {Pedram} and K. M. {Chugg} and B. {Peter}}, 
journal={IEEE Transactions on Computers}, 
title={Pre-defined Sparsity for Low-Complexity Convolutional Neural Networks}, 
year={2020}, 
volume={}, number={}, pages={1-1},}


and 

Conference:
===========

@INPROCEEDINGS{8919683, author={S. {Kundu} and S. {Prakash} and H. {Akrami} and P. A. {Beerel} and K. M. {Chugg}}, 
booktitle={2019 57th Annual Allerton Conference on Communication, Control, and Computing (Allerton)}, 
title={pSConv: A Pre-defined Sparse Kernel Based Convolution for Deep CNNs}, 
year={2019}, volume={}, number={}, pages={100-107},}


Abstract:
The high energy cost of processing deep convolutional neural networks impedes their ubiquitous deployment in energy-constrained platforms such as embedded systems and IoT devices. This work introduces convolutional layers with pre-defined sparse 2D kernels that have support sets that repeat periodically within and across filters. Due to the efficient storage of our periodic sparse kernels, the parameter savings can translate into considerable improvements in energy efficiency due to reduced DRAM accesses, thus promising significant improvements in the trade-off between energy consumption and accuracy for both training and inference. To evaluate this approach, we performed experiments with two widely accepted datasets, CIFAR-10 and Tiny ImageNet in sparse variants of the ResNet18 and VGG16 architectures. Compared to baseline models, our proposed sparse variants require up to ∼82% fewer model parameters with 5.6× fewer FLOPs with negligible loss in accuracy for ResNet18 on CIFAR-10. For VGG16 trained on Tiny ImageNet, our approach requires 5.8× fewer FLOPs and up to ∼83.3% fewer model parameters with a drop in top-5 (top-1) accuracy of only 1.2%(∼2.1%). We also compared the performance of our proposed architectures with that of ShuffleNet and MobileNetV2. Using similar hyperparameters and FLOPs, our ResNet18 variants yield an average accuracy improvement of ∼2.8%.

A version of our work is also available at the arxiv link: https://arxiv.org/abs/2001.10710