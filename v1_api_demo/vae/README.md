#Variational Autoencoder (VAE)

This demo implements VAE training described in the original paper (https://arxiv.org/abs/1312.6114).


In order to run the model, first download the MNIST dataset by running the shell script in ./data.

Then you can run the command below. The flag --useGpu specifies whether to use gpu for training (0 is cpu, 1 is gpu).  

$python vae_train.py [--use_gpu 1]

The generated images will be stored in ./samples/
The corresponding models will be stored in ./params/
