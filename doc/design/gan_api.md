'''
GAN implementation, just a demo.
'''
```python
# pd for short, should be more concise.
from paddle.v2 as pd
import numpy as np
import logging

X = pd.data(pd.float_vector(784))
```
# Conditional-GAN should be a class. 
### Class member function: the initializer.
```python
class DCGAN(object):
  def __init__(self, y_dim=None):
  
    # hyper parameters  
    self.y_dim = y_dim # conditional gan or not
    self.batch_size = 100
    self.z_dim = z_dim # input noise dimension

    # define parameters of discriminators
    self.D_W0 = pd.Variable(shape=[784, 128], data=pd.gaussian_normal_randomizer())
    self.D_b0 = pd.Variable(np.zeros(128)) # variable also support initialization using a  numpy data
    self.D_W1 = pd.Variable(shape=[784, 128], data=pd.gaussian_normal_randomizer())
    self.D_b1 = pd.Variable(np.zeros(128)) # variable also support initialization using a  numpy data
    self.D_W2 = pd.Varialble(np.random.rand(128, 1))
    self.D_b2 = pd.Variable(np.zeros(128))
    self.theta_D = [self.D_W0, self.D_b0, self.D_W1, self.D_b1, self.D_W2, self.D_b2]

    # define parameters of generators
    self.G_W0 = pd.Variable(shape=[784, 128], data=pd.gaussian_normal_randomizer())
    self.G_b0 = pd.Variable(np.zeros(128)) # variable also support initialization using a  numpy data
    self.G_W1 = pd.Variable(shape=[784, 128], data=pd.gaussian_normal_randomizer())
    self.G_b1 = pd.Variable(np.zeros(128)) # variable also support initialization using a  numpy data
    self.G_W2 = pd.Varialble(np.random.rand(128, 1))
    self.G_b2 = pd.Variable(np.zeros(128))
    self.theta_G = [self.G_W0, self.G_b0, self.G_W1, self.G_b1, self.G_W2, self.G_b2]
```

### Class member function: Generator Net
```python
def generator(self, z, y = None):

    # Generator Net
    if not self.y_dim:
      z = pd.concat(1, [z, y])
      
    G_h0 = pd.fc(z, self.G_w0, self.G_b0)
    G_h0_bn = pd.batch_norm(G_h0)
    G_h0_relu = pd.relu(G_h0_bn)
    
    G_h1 = pd.fc(G_h0_relu, self.G_w1, self.G_b1)
    G_h1_bn = pd.batch_norm(G_h1)
    G_h1_relu = pd.relu(G_h1_bn)
    
    G_h2 = pd.deconv(G_h1_relu, self.G_W2, self.G_b2))
    G_im = pd.tanh(G_im)
    return G_im
```

### Class member function: Discriminator Net
```python
def discriminator(self, image):

    # Discriminator Net
    D_h0 = pd.conv2d(image, self.D_w0, self.D_b0)
    D_h0_bn = pd.batchnorm(h0)
    D_h0_relu = pd.lrelu(h0_bn)
    
    D_h1 = pd.conv2d(D_h0_relu, self.D_w1, self.D_b1)
    D_h1_bn = pd.batchnorm(D_h1)
    D_h1_relu = pd.lrelu(D_h1_bn)
    
    D_h2 = pd.fc(D_h1_relu, self.D_w2, self.D_b2)
    return D_h2
```

### Class member function: Build the model
```python
def build_model(self):

    # input data
    if self.y_dim:
        self.y = pd.data(pd.float32, [self.batch_size, self.y_dim])
    self.images = pd.data(pd.float32, [self.batch_size, self.im_size, self.im_size])
    self.faked_images = pd.data(pd.float32, [self.batch_size, self.im_size, self.im_size])
    self.z = pd.data(tf.float32, [None, self.z_size])
    
    # if conditional GAN
    if self.y_dim:
      self.G = self.generator(self.z, self.y)
      self.D_t = self.discriminator(self.images)
      # generated fake images
      self.sampled = self.sampler(self.z, self.y)
      self.D_f = self.discriminator(self.images)
    else: # original version of GAN
      self.G = self.generator(self.z)
      self.D_t = self.discriminator(self.images)
      # generate fake images
      self.sampled = self.sampler(self.z)
      self.D_f = self.discriminator(self.images)
    
    self.d_loss_real = pd.reduce_mean(pd.cross_entropy(self.D_t, np.ones(self.batch_size))
    self.d_loss_fake = pd.reduce_mean(pd.cross_entropy(self.D_f, np.zeros(self.batch_size))
    self.d_loss = self.d_loss_real + self.d_loss_fake
    
    self.g_loss = pd.reduce_mean(pd.cross_entropy(self.D_f, np.ones(self.batch_szie))
```

# Main function for the demo:
```python
if __name__ == "__main__":

    # dcgan
    dcgan = DCGAN()
    dcgan.build_model()

    # load mnist data
    data_X, data_y = self.load_mnist()
    
    # Two subgraphs required!!!
    d_optim = pd.train.Adam(lr = .001, beta= .1).minimize(self.d_loss, )
    g_optim = pd.train.Adam(lr = .001, beta= .1).minimize(self.g_loss)

    # executor
    sess = pd.executor()
    
    # training
    for epoch in xrange(10000):
      for batch_id in range(N / batch_size):
        idx = ...
        # sample a batch
        batch_im, batch_label = data_X[idx:idx+batch_size], data_y[idx:idx+batch_size]
        # sample z
        batch_z = np.random.uniform(-1., 1., [batch_size, z_dim])

        if batch_id % 2 == 0:
          sess.eval(d_optim, 
                   feed_dict = {dcgan.images: batch_im,
                                dcgan.y: batch_label,
                                dcgan.z: batch_z})
        else:
          sess.eval(g_optim,
                   feed_dict = {dcgan.z: batch_z})
```
