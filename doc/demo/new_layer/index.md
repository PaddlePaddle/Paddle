Writing New Layers
----------
This tutorial will guide you to write customized layers in PaddlePaddle. We will utilize fully connected layer as an example to guide you through the following steps for writing a new layer.
- Derive equations for the forward and backward part of the layer.
- Implement C++ class for the layer.
- Implement Python Wrapper for the layer.

## Derive Equations
First we need to derive equations of the *forward* and *backward* part of the layer. The forward part computes the output given an input. The backward part computes the gradients of the input and the parameters given the the gradients of the output.

The illustration of a fully connected layer is shown in the following figure. In a fully connected layer, all output nodes are connected to all the input nodes.
<center> ![](./FullyConnected.jpg) </center>

The *forward part* of a layer transforms an input into the corresponding output.
Fully connected layer takes a dense input vector with dimension $D_i$. It uses a transformation matrix $W$ with size $D_i \times D_o$ to project x into a $D_o$ dimensional vector, and add a bias vector  $b$ with dimension $D_o$ to the vector.
\[y = f(W^T x + b) \]
where $f(.)$ is an nonlinear *activation* function, such as sigmoid, tanh, and Relu.

The transformation matrix $W$ and bias vector $b$ are the *parameters* of the layer. The *parameters* of a layer are learned during training in the *backward pass*. The backward pass computes the gradients of the output function with respect to all parameters and inputs. The optimizer can use chain rule to compute the gradients of the loss function with respect to each parameter. Suppose our loss function is $c(y)$, then
\[\frac{\partial c(y)}{\partial x} = \frac{\partial c(y)}{\partial y} \frac{\partial y}{\partial x} \]

Suppose $z = f(W^T x + b)$, then
\[ \frac{\partial y}{\partial z} = \frac{\partial f(z)}{\partial z}\]
 This derivative can be automatically computed by our base layer class.

Then, for fully connected layer, we need to compute $\frac{\partial z}{\partial x}$, and $\frac{\partial z}{\partial W}$, and $\frac{\partial z}{\partial b}$.
\[ \frac{\partial z}{\partial x} = W \]
\[ \frac{\partial z_j}{\partial W_{ij}} = x_i \]
\[ \frac{\partial z}{\partial b} = \mathbf 1 \]
where $\mathbf 1$ is an all one vector, $W_{ij}$ is the number at the i-th row and j-th column of the matrix $W$, $z_j$ is the j-th component of the vector $z$, and $x_i$ is the i-th component of the vector $x$.

Then we can use chain rule to calculate $\frac{\partial z}{\partial x}$, and $\frac{\partial z}{\partial W}$. The details of the computation will be given in the next section.

## Implement C++ Class
The C++ class of the layer implements the initialization, forward, and backward part of the layer. The fully connected layer is at `paddle/gserver/layers/FullyConnectedLayer.h` and `paddle/gserver/layers/FullyConnectedLayer.cpp`. We list simplified version of the code below.

It needs to derive the base class `paddle::BaseLayer`, and it needs to override the following functions:

- constructor and destructor.
- `init` function. It is used to initialize the parameters and settings.
- `forward`. It implements the forward part of the layer.
- `backward`. It implements the backward part of the layer.
- `prefetch`. It is utilized to determine the rows corresponding parameter matrix to prefetch from parameter server. You do not need to override this function if your layer does not need remote sparse update. (most layers do not need to support remote sparse update)


The header file is listed below:

```C
namespace paddle {
/**
 * A layer has full connections to all neurons in the previous layer.
 * It computes an inner product with a set of learned weights, and
 * (optionally) adds biases.
 *
 * The config file api is fc_layer.
 */

class FullyConnectedLayer : public Layer {
protected:
  WeightList weights_;
  std::unique_ptr<Weight> biases_;

public:
  explicit FullyConnectedLayer(const LayerConfig& config)
      : Layer(config) {}
  ~FullyConnectedLayer() {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  Weight& getWeight(int idx) { return *weights_[idx]; }

  void prefetch();
  void forward(PassType passType);
  void backward(const UpdateCallback& callback = nullptr);
};
}  // namespace paddle
```

It defines the parameters as class variables. We use `Weight` class as abstraction of parameters. It supports multi-thread update. The details of this class will be described in details in the implementations.
- `weights_` is a list of weights for the transformation matrices. The current implementation can have more than one inputs. Thus, it has a list of weights. One weight corresponds to an input.
- `biases_` is a weight for the bias vector.

The fully connected layer does not have layer configuration hyper-parameters. If there are some layer hyper-parameters, a common practice is to store it in `LayerConfig& config`, and put it into a class variable in the constructor.

The following code snippet implements the `init` function.
- First, every `init` function must call the `init` function of the base class `Layer::init(layerMap, parameterMap);`. This statement will initialize the required variables and connections for each layer.
- The it initializes all the weights matrices $W$. The current implementation can have more than one inputs. Thus, it has a list of weights.
- Finally, it initializes the bias.


```C
bool FullyConnectedLayer::init(const LayerMap& layerMap,
                               const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  /* initialize the weightList */
  CHECK(inputLayers_.size() == parameters_.size());
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    // Option the parameters
    size_t height = inputLayers_[i]->getSize();
    size_t width = getSize();

    // create a new weight
    if (parameters_[i]->isSparse()) {
      CHECK_LE(parameters_[i]->getSize(), width * height);
    } else {
      CHECK_EQ(parameters_[i]->getSize(), width * height);
    }
    Weight* w = new Weight(height, width, parameters_[i]);

    // append the new weight to the list
    weights_.emplace_back(w);
  }

  /* initialize biases_ */
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
  }

  return true;
}

```


The implementation of the forward part has the following steps.
- Every layer must call `Layer::forward(passType);` at the beginning of its `forward` function.
- Then it allocates memory for the output using `reserveOutput(batchSize, size);`. This step is necessary because we support the batches to have different batch sizes. `reserveOutput` will change the size of the output accordingly. For the sake of efficiency, we will allocate new memory if we want to expand the matrix, but we will reuse the existing memory block if we want to shrink the matrix.
- Then it computes $\sum_i W_i x + b$ using Matrix operations. `getInput(i).value` retrieve the matrix of the i-th input. Each input is a $batchSize \times dim$ matrix, where each row represents an single input in a batch. For a complete lists of supported matrix operations, please refer to `paddle/math/Matrix.h` and `paddle/math/BaseMatrix.h`.
- Finally it applies the activation function using `forwardActivation();`. It will automatically applies the corresponding activation function specifies in the network configuration.


```C
void FullyConnectedLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  int batchSize = getInput(0).getBatchSize();
  int size = getSize();

  {
    // Settup the size of the output.
    reserveOutput(batchSize, size);
  }

  MatrixPtr outV = getOutputValue();

  // Apply the the transformation matrix to each input.
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    auto input = getInput(i);
    CHECK(input.value) << "The input of 'fc' layer must be matrix";
    i == 0 ? outV->mul(input.value, weights_[i]->getW(), 1, 0)
           : outV->mul(input.value, weights_[i]->getW(), 1, 1);
  }

  /* add the bias-vector */
  if (biases_.get() != NULL) {
    outV->addBias(*(biases_->getW()), 1);
  }

  /* activation */ {
    forwardActivation();
  }
}
```

The implementation of the backward part has the following steps.
- ` backwardActivation();` computes the gradients of the activation. The gradients will be multiplies in place to the gradients of the output, which can be retrieved using `getOutputGrad()`.
- Compute the gradients of bias. Notice that we an use `biases_->getWGrad()` to get the gradient matrix of the corresponding parameter. After the gradient of one parameter is updated, it *MUST* call `getParameterPtr()->incUpdate(callback);`. This is utilize for parameter update over multiple threads or multiple machines.
- Then it computes the gradients of the transformation matrices and inputs, and it calls `incUpdate` for the corresponding parameter. This gives the framework the chance to know whether it has gathered all the gradient to one parameter so that it can do some overlapping work (e.g., network communication)


```C
void FullyConnectedLayer::backward(const UpdateCallback& callback) {
  /* Do derivation for activations.*/ {
    backwardActivation();
  }

  if (biases_ && biases_->getWGrad()) {
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);

    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }

  bool syncFlag = hl_get_sync_flag();

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    /* Calculate the W-gradient for the current layer */
    if (weights_[i]->getWGrad()) {
      MatrixPtr input_T = getInputValue(i)->getTranspose();
      MatrixPtr oGrad = getOutputGrad();
      {
        weights_[i]->getWGrad()->mul(input_T, oGrad, 1, 1);
      }
    }


    /* Calculate the input layers error */
    MatrixPtr preGrad = getInputGrad(i);
    if (NULL != preGrad) {
      MatrixPtr weights_T = weights_[i]->getW()->getTranspose();
      preGrad->mul(getOutputGrad(), weights_T, 1, 1);
    }

    {
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
  }
}
```

The `prefetch` function specifies the rows that need to be fetched from parameter server during training. It is only useful for remote sparse training. In remote sparse training, the full parameter matrix is stored distributedly at the parameter server. When the layer uses a batch for training, only a subset of locations of the input is non-zero in this batch. Thus, this layer only needs the rows of the transformation matrix corresponding to the locations of these non-zero entries. The `prefetch` function specifies the ids of these rows.

Most of the layers do not need remote sparse training function. You do not need to override this function in this case.

```C
void FullyConnectedLayer::prefetch() {
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    auto* sparseParam =
        dynamic_cast<SparsePrefetchRowCpuMatrix*>(weights_[i]->getW().get());
    if (sparseParam) {
      MatrixPtr input = getInputValue(i);
      sparseParam->addRows(input);
    }
  }
}
```

Finally, you can use `REGISTER_LAYER(fc, FullyConnectedLayer);` to register the layer. `fc` is the identifier of the layer, and `FullyConnectedLayer` is the class name of the layer.

```C
namespace paddle {
REGISTER_LAYER(fc, FullyConnectedLayer);
}
```

If the `cpp` file is put into `paddle/gserver/layers`, it will be automatically compiled.

## Implement Python Wrapper
Implementing Python wrapper allows us to use the added layer in configuration files. All the Python wrappers are in file `python/paddle/trainer/config_parser.py`. An example of the Python wrapper for fully connected layer is listed below. It has the following steps:
- Use `@config_layer('fc’)` at the decorator for all the Python wrapper class. `fc` is the identifier of the layer.
- Implements `__init__` constructor function.
	- It first call  `super(FCLayer, self).__init__(name, 'fc', size, inputs=inputs, **xargs)` base constructor function. `FCLayer` is the Python wrapper class name, and `fc` is the layer identifier name. They must be correct in order for the wrapper to work.
	- Then it computes the size and format (whether sparse) of each transformation matrix as well as the size.

```python
@config_layer('fc')
class FCLayer(LayerBase):
    def __init__(
            self,
            name,
            size,
            inputs,
            bias=True,
            **xargs):
        super(FCLayer, self).__init__(name, 'fc', size, inputs=inputs, **xargs)
        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            psize = self.config.size * input_layer.size
            dims = [input_layer.size, self.config.size]
            format = self.inputs[input_index].format
            sparse = format == "csr" or format == "csc"

            if sparse:
                psize = self.inputs[input_index].nnz

            self.create_input_parameter(input_index, psize, dims, sparse, format)
        self.create_bias_parameter(bias, self.config.size)
```

In network configuration, the layer can be specifies using the following code snippets. The arguments of this class are:
- `name` is the name identifier of the layer instance.
- `type` is the type of the layer, specified using layer identifier.
- `size` is the output size of the layer.
- `bias` specifies whether this layer instance has bias.
- `inputs` specifies a list of layer instance names as inputs.

```python
Layer(
    name = "fc1",
    type = "fc",
    size = 64,
    bias = True,
    inputs = [Input("pool3")]
)
```

You are also recommended to implement a helper for the Python wrapper, which makes it easier to write models. You can refer to `python/paddle/trainer_config_helpers/layers.py` for examples.
