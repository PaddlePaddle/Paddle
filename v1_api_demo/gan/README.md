# Generative Adversarial Networks (GAN) 

This demo implements GAN training described in the original GAN paper (https://arxiv.org/abs/1406.2661) and DCGAN (https://arxiv.org/abs/1511.06434).

The general training procedures are implemented in gan_trainer.py. The neural network configurations are specified in gan_conf.py (for synthetic data) and gan_conf_image.py (for image data).

In order to run the model, first download the corresponding data by running the shell script in ./data.
Then you can run the command below. The flag -d specifies the training data (cifar, mnist or uniform) and flag --useGpu specifies whether to use gpu for training (0 is cpu, 1 is gpu).  

$python gan_trainer.py -d cifar --use_gpu 1

The generated images will be stored in ./cifar_samples/
The corresponding models will be stored in ./cifar_params/
