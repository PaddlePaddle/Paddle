import paddle.v2 as pd

def build_descrimitor(x):
    '''
    descrimitor should be called twice, so make it a function.
    '''
    with pd.block('descrimitor') as D:
        # parameters for D
        W1 = pd.Variable('W1', shape=[784, 128], data=pd.gaussian_normal_randomizer())
        b1 = pd.Variable('b1', np.zeros(128)) # variable also support initialization using a  numpy data
        W2 = pd.Varialble('W2', np.random.rand(128, 1))
        b2 = pd.Variable('b2', np.zeros(128))

        h1 = pd.relu(pd.matmul(x, W1) + b1)
        fake = pd.matmul(h1, w2) + b2
        prob = pd.sigmoid(logit)
        return prob, fake

def build_generator(x):
    with pd.block('generator') as G:
        Z = pd.data(pd.float_vector(100))

        W1 = pd.Variable('W1', shape=[784, 128], data=pd.gaussian_normal_randomizer())
        b1 = pd.Variable('b1', np.zeros(128)) # variable also support initialization using a  numpy data
        W2 = pd.Varialble('W2', np.random.rand(128, 1))
        b2 = pd.Variable('b2', np.zeros(128))

        # build network
        h1 = pd.relu(pd.matmul(z, W1) + b1)
        log_prob = pd.matmul(h1, W2) + b2
        fake = pd.sigmoid(log_prob)

        return fake


# build a network
with pd.block('discriminator') as D:
    X = pd.data(pd.float_vector(784))
    real_prob, real_image = build_descrimitor(X)

    Z = pd.data(pd.float_vector(100))
    G_sample = build_generator(Z)

    fake_prob, fake_image = build_descrimitor(G_sample)

    # a mini-batch of 1. as probability 100%
    ones_label = pd.data(pd.float_vector(1))
    # a mini-batch of 0. as probability 0%
    zeros_label = pd.data(pd.float_vector(1))

    loss_real = pd.reduce_mean(pd.cross_entropy(data=real_prob, label=ones_label))
    loss_fake = pd.reduce_mean(pd.cross_entropy(data=real_fake, label=zeros_label))
    loss = pd.add_two(loss_real, loss_fake)

    optimizer = pd.optimier([loss])


block = pd.block('discriminator')
block.execute()
