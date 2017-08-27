# Namespace Design

## Namespace vs Scope

## Namespace
a global scope with name prefix to demonstrate a solf scope.
```python
import paddle as pd

# decalre global variable to store results from functions
prob = pd.Variable()
sample = pd.Variable()

def descriminator(x):
    with pd.scope():
        prob_ = xxxop(x)
        pd.assign(prob_)

def generator(z):
    with pd.namespace():
        sample_ = xxxop(z)
        pd.assign(sample_)

image = pd.data('image')
z = pd.data('z')

prob_real = descriminator(image)
sample = generator(z)
prob_fake = descriminator(sample)
```

## Scope

```python
import paddle as pd

def descriminator(x):
    with pd.namespace('descriminator'):
        prob = xxxop(x)
        return prob

def generator(z):
    with pd.namespace('generator'):
        sample = xxxop(z)
        return sample

image = pd.data('image')
z = pd.data('z')

prob_real = descriminator(image)
sample = generator(z)
prob_fake = descriminator(sample)


```
