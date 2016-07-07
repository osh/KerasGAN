# KerasGAN
This module includes a GAN implementation in Keras for the MNIST data set
See full article @ https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/

# GAN Overview
The GAN includes a generative and discrimintive network defined in Keras' functional API, they can then be chained together to make a composite model for training end-to-end.
![GAN BlockDiag](https://oshearesearch.com/wp-content/uploads/2016/07/mnist_gan.png)

# Generated Images
Generated Images aren't perfect, the network is still pretty small and additional tuning would likely help.
![Generated Digits](https://oshearesearch.com/wp-content/uploads/2016/07/mnist_gan7-300x300.png)

# Learning Rates
I tend to find the having a larger (faster) learning rate on the discrimintive model leads to better results than keeping them equal in the discriminitive and generative training tasks. 
Would be curious to hear from others who are familiar with GAN tuning here.
When training with imbalanced learning rates like this, discriminitive loss stays pretty low, and the discriminitive model generally stays ahead of discriminatring new strange represenentations from the generative model.
![Training Loss](https://oshearesearch.com/wp-content/uploads/2016/07/mnist_gan_loss4.png)
