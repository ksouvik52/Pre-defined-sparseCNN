## Pre-Defined Sparsity for Low Complexity CNNs
<p align="center"><img width="30%" src="/Images/IEEE_TC.jpg"/><img width="20%" src="/Images/allerton2019-350.jpg"/></p><br/> 

This repository constains official Pytorch implementation of the paper, titled [Pre-defined Sparsity for Low-Complexity Convolutional Neural Networks](https://ieeexplore.ieee.org/document/8988206) which is published as a journal paper in **IEEE transactions on Computer, 2020**. The work is extended version of our conference paper [pSConv: A Pre-defined Sparse Kernel Based Convolution for Deep CNNs](https://ieeexplore.ieee.org/document/8919683) published in **Allerton Conference , 2019**. Currently the repo contains example code for `VGG16` model on `CIFAR-10` classification task. 

### Abstract
The high energy cost of processing deep convolutional neural networks impedes their ubiquitous deployment in energy-constrained platforms such as embedded systems and IoT devices. This article introduces convolutional layers with pre-defined sparse 2D kernels that have support sets that repeat periodically within and across filters. Due to the efficient storage of our periodic sparse kernels, the parameter savings can translate into considerable improvements in energy efficiency due to reduced DRAM accesses, thus promising significant improvements in the trade-off between energy consumption and accuracy for both training and inference. To evaluate this approach, we performed experiments with two widely accepted datasets, CIFAR-10 and Tiny ImageNet in sparse variants of the ResNet18 and VGG16 architectures. Compared to baseline models, our proposed sparse variants require up to ∼82% fewer model parameters with 5.6× fewer FLOPs with negligible loss in accuracy for ResNet18 on CIFAR-10. For VGG16 trained on Tiny ImageNet, our approach requires 5.8× fewer FLOPs and up to ∼83.3% fewer model parameters with a drop in top-5 (top-1) accuracy of only 1.2% ( ∼2.1% ). We also compared the performance of our proposed architectures with that of ShuffleNet and MobileNetV2. Using similar hyperparameters and FLOPs, our ResNet18 variants yield an average accuracy improvement of ∼2.8% .

### Proposed CNN model and results  
<p align="center"><img width="20%" src="/Images/periodic_sparse_dense_conv.png" /><img width="70%"  src="/Images/ShuffleNet_MobileNetV2_compare_acc_flops_cifar_tiny.png" /><img width="20%" src="/Images/VGG_Tiny_ImageNet.png" /><img width="20%" src="/Images/VGG_CIFAR-10.png" /><img width="20%" src="/Images/Res_Tiny_ImageNet.png" /><img width="20%" src="/Images/Res_CIFAR-10.png" /></p><br/> 

### Requirements
All experiments were conducted in AWS p3.2x large instances (Nvidia V100 GPU) with Pytorch `v1.3` and CUDA `v9.2`.

### Cite this work
If you find this project useful to you, please cite our work:

      @misc
      {8988206, 
      author  ={S. {Kundu} and M. {Nazemi} and M. {Pedram} and K. M. {Chugg} and P. A. {Beerel}}, 
      journal ={IEEE Transactions on Computers}, 
      title   ={Pre-defined Sparsity for Low-Complexity Convolutional Neural Networks}, 
      volume={69},
      number={7},
      pages={1045-1058},}
and 

      @misc
      {8919683, 
      author    ={S. {Kundu} and S. {Prakash} and H. {Akrami} and P. A. {Beerel} and K. M. {Chugg}}, 
      booktitle ={2019 57th Annual Allerton Conference on Communication, Control, and Computing (Allerton)}, 
      title     ={pSConv: A Pre-defined Sparse Kernel Based Convolution for Deep CNNs}, 
      year      ={2019}, 
      pages     ={100-107},}

 arxiv version link: https://arxiv.org/abs/2001.10710
