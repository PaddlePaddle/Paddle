'''
A demo with multiple model.
'''
# pd for short
import paddle.v2 as pd

D_model = Model()
G_model = Model()

# NOTE found that the functional way is pain to define two model with the same operators.
# duplicate operators' copy is necessary.

# to make it possible to share the parameters across different models, the parameters are
# defined in global scope.
D_W1 = pd.Variable(shape=[784, 128], data=pd.gaussian_normal_randomizer())
D_b1 = pd.Variable(np.zeros(128)) # variable also support initialization using a  numpy data
D_W2 = pd.Varialble(np.random.rand(128, 1))
D_b2 = pd.Variable(np.zeros(128))
# Discriminator's parameters
theta_D = [D_W1, D_b1, D_W2, D_b2]

def create_D_model(model):
    '''
    create Descrimator model.
    '''
    X = model.data(pd.float_vector(784))
    D_h1 = model.relu(pd.matmul(x, D_W1) + D_b1)
    fake = model.matmul(D_h1, D_w2) + D_b2
    D_prob = model.sigmoid(D_logit)
    return D_prob, fake


def create_G_model(model):
    '''
    create Generator model.
    '''
    # NOTE this line is duplicate between both D and G
    X = D_model.data(pd.float_vector(784))


