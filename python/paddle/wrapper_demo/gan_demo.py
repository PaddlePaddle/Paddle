'''
GAN implementation, just a demo.
'''
# pd for short, should be more concise.
from paddle.v2 as pd
import numpy as np
import logging

X = pd.data(pd.float_vector(784))

# Discriminator Net
# define parameters
D_W1 = pd.Variable(shape=[784, 128], data=pd.gaussian_normal_randomizer())
D_b1 = pd.Variable(np.zeros(128)) # variable also support initialization using a  numpy data
D_W2 = pd.Varialble(np.random.rand(128, 1))
D_b2 = pd.Variable(np.zeros(128))

# Discriminator's parameters
theta_D = [D_W1, D_b1, D_W2, D_b2]


# Generator Net
Z = pd.data(pd.float_vector(100))

G_W1 = pd.Variable(shape=[784, 128], data=pd.gaussian_normal_randomizer())
G_b1 = pd.Variable(np.zeros(128)) # variable also support initialization using a  numpy data
G_W2 = pd.Varialble(np.random.rand(128, 1))
G_b2 = pd.Variable(np.zeros(128))

theta_G = [G_W1, G_W2, G_b1, G_b2]

def sample_Z(m, n):
    return np.random.uniform(-1, 1., size=[m, n])

def generator(z):
    G_h1 = pd.relu(pd.matmul(z, G_W1) + G_b1)
    G_log_prob = pd.matmul(G_h1, G_W2) + G_b2
    G_prob = pd.sigmoid(G_log_prob)
    return G_prob

def discriminator(x):
    D_h1 = pd.relu(pd.matmul(x, D_W1) + D_b1)
    fake = pd.matmul(D_h1, D_w2) + D_b2
    D_prob = pd.sigmoid(D_logit)
    return D_prob, fake

# a mini-batch of 1. as probability 100%
ones_label = pd.data(pd.float_vector(1))
# a mini-batch of 0. as probability 0%
zeros_label = pd.data(pd.float_vector(1))

# model config
G_sample = generator(Z)
D_real_prob, D_real_image = discriminator(X)
D_fake_prob, D_fake_image = discriminator(G_sample)

D_loss_real = pd.reduce_mean(pd.cross_entropy(data=D_real_prob, label=ones_label))
D_loss_fake = pd.reduce_mean(pd.cross_entropy(data=D_real_fake, label=zeros_label))
D_loss = D_loss_real + D_loss_fake

G_loss = pd.reduce_mean(pd.cross_entropy(data=D_loss_fake, label=ones_label))

D_solver = pd.optimizer.Adam().minimize(D_loss, var_list=theta_D)
G_solver = pd.optimizer.Adam().minimize(G_loss, var_list=theta_G)

# init all parameters
initializer = pd.variable_initialzier()
# also ok: initializer = pd.variable_initialzier(vars=theta_D+theta_G)
pd.eval(initializer)

def data_provier(path=...):
    # ...
    yield batch

for i in range(10000):
    for batch_no, batch in enumerate(data_provider('train_data.txt')):
        # train Descrimator first
        _, D_loss_cur = pd.eval([D_solver, D_loss], feed_dict={X:batch, Z:sample_Z(batch.size, 10)})
        # get Generator's fake samples
        samples = pd.eval(G_sample, feed={Z: sample_Z(16, 100)})
        # train Generator latter
        _, G_loss_cur = pd.eval([G_solver, G_loss], feed_)

        if batch_no % 100:
            logger.info("batch %d, D loss: %f" % (batch_no, D_loss_cur))
            logger.info("batch %d, G loss: %f" % (batch_no, G_loss_cur))
