'''
GAN implementation, just a demo.
'''
# pd for short of paddle, should be more concise.
# all paddle.layer and paddle.op are merge into paddle namespace, that means
# we can use `pd.fc` and `pd.add` to use fc layer and add operator.
from paddle.v2 as pd
import numpy as np
import logging

# hard to write a GAN using V2's native syntax, implementation with some hack.

# paddle.init
pd.init(use_gpu=False, trainer_count=1)

# NOTE here pd.data is short of paddle.layer.data in V2
# NOTE `name` is not necessary, but also supported
X = pd.data(name='X', pd.float_vector(784))

# NOTE In V2, all model parameters are created by layers
# GAN cannot be implemented using native V2, but all the model based on V2 syntax
# will hide the following parameters in layer, and that is easy to suppored using
# new op-based syntax features.

# Discriminator Net
# define parameters
D_W1 = pd.Variable(shape=[784, 128], data=pd.gaussian_normal_randomizer())
D_b1 = pd.Variable(np.zeros(128)) # variable also support initialization using a  numpy data
D_W2 = pd.Varialble(np.random.rand(128, 1))
D_b2 = pd.Variable(np.zeros(128))

# Discriminator's parameters
theta_D = [D_W1, D_b1, D_W2, D_b2]


# Generator Net
Z = pd.data(name='Z', pd.float_vector(100))

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
ones_label = pd.data(name='ones_label', type=pd.float_vector(1))
# a mini-batch of 0. as probability 0%
zeros_label = pd.data(name='zeros_label', type=pd.float_vector(1))

# model config
G_sample = generator(Z)
D_real_prob, D_real_image = discriminator(X)
D_fake_prob, D_fake_image = discriminator(G_sample)

D_loss_real = pd.reduce_mean(pd.cross_entropy(data=D_real_prob, label=ones_label))
D_loss_fake = pd.reduce_mean(pd.cross_entropy(data=D_real_fake, label=zeros_label))
D_loss = D_loss_real + D_loss_fake

G_loss = pd.reduce_mean(pd.cross_entropy(data=D_loss_fake, label=ones_label))

# NOTE can V2's optimizer express `maxmize`?
D_solver = pd.optimizer.Adam(
    learning_rate=1e-3,
    regularization=pd.L2Regularization(rate=1e-3),
    model_average=pd.ModelAverage(average_window=0.5),
    # NOTE `vars` is a new argument added to v2 optimizer
    vars=theta_D)

G_solver = pd.optimizer.Adam().minimize(G_loss, var_list=theta_G)

# init all parameters
initializer = pd.variable_initialzier()
# also ok: initializer = pd.variable_initialzier(vars=theta_D+theta_G)
#pd.eval(initializer)

def data_provier(path=...):
    # ...
    yield batch

data_reader = pd.batch(
    pd.reader.shuffle(data_provider, buf_size=1000),
    batch_size=1000)


D_trainer = pd.trainer.SGD(
    cost=D_loss,
    parameters=D_theta,
    update_equation=D_solver)

G_trainer = pd.trainer.SGD(
    cost=G_loss,
    parameters=G_theta,
    update_equation=G_solver)

def event_handler(event):
    # ...

cur_batch = None

def cur_batch_reader():
    '''
    just read one batch, make it possible that D ang G get the same image when trained.
    '''
    global cur_batch
    if cur_batch is None:
        cur_batch = data_reader().next()
    yield cur_batch

for i in range(1000):
    # Train Descrimator first, with just a batch
    D_trainer.train(
        reader = cur_batch_reader,
        event_handler=event_handler,
        feeding={'image': 0, 'ones_label': 1, 'zeros_label': 2},
        # NOTE just run 1 pass here
        num_passes=1)

    # train Generator latter, with the same batch of data
    G_trainer.train(
        reader = cur_batch_reader,
        event_handler=event_handler,
        feeding={'image': 0, 'ones_label': 1, 'zeros_label': 2},
        # NOTE just run 1 pass here
        num_passes=1)

    # proceed to next batch of data
    cur_batch = data_reader().next()
