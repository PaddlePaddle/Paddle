## Auto Gradient Check Design

## Background：
- Generally, it is easy to check whether the forward computation of an Operator is correct or not. However, backpropagation is a notoriously difficult algorithm to debug and get right because of the following challenges:
  1. The formula for backpropagation formula should be correct according to the forward computation.
  2. The Implementation of the above shoule be correct in CPP.
  3. It is difficult to prepare an unbiased test data.

- Auto gradient checking gets a numerical gradient using forward Operator and uses it as a reference for the backward Operator's result. It has several advantages:
  1. Numerical gradient checker only needs the forward operator.
  2. The user only needs to prepare the input data for forward Operator and not worry about the backward Operator.

## Mathematical Theory
The following documents from Stanford have a detailed explanation of how to compute the numerical gradient and why it is useful.

- [Gradient checking and advanced optimization(en)](http://deeplearning.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization)
- [Gradient checking and advanced optimization(cn)](http://ufldl.stanford.edu/wiki/index.php/%E6%A2%AF%E5%BA%A6%E6%A3%80%E9%AA%8C%E4%B8%8E%E9%AB%98%E7%BA%A7%E4%BC%98%E5%8C%96)


## Numerical Gradient Implementation
### Python Interface
```python
def get_numerical_gradient(op,
                         input_values,
                         output_name,
                         input_to_check,
                         delta=0.005,
                         local_scope=None):
    """
    Get Numerical Gradient for the input of an operator.

    :param op: C++ operator instance, could be an network.
    :param input_values: The input variables. Should be an dictionary, whose key is
    variable name, and value is a numpy array.
    :param output_name: The final output variable name.
    :param input_to_check: The input variable with respect to which the gradient has to be computed.
    :param delta: The perturbation value for numerical gradient method. The
    smaller the delta, the more accurate the result. But if the delta is too
    small, it will suffer from the numerical stability problem.
    :param local_scope: The local scope used for get_numeric_gradient.
    :return: The gradient array in numpy format.
    """
```

### Explanation:

- Why do we need an `output_name`
  - An Operator may have multiple Outputs, one can compute an independent gradient from each Output. So the caller should specify the name of the output variable.

- Why do we need `input_to_check`
  - One operator can have multiple inputs. Gradient Op can calculate the gradient of these inputs at the same time. But Numerical Gradient needs to calculate them one by one. So `get_numeric_gradient` is designed to calculate the gradient for one input. If you need to compute multiple inputs, you can call `get_numeric_gradient` multiple times each with a different input.


### Core Algorithm Implementation


```python
    # we only compute the gradient of one element a time.
    # we use a for loop to compute the gradient of each element.
    for i in xrange(tensor_size):
        # get one input element using the index i.
        original = tensor_to_check.get_float_element(i)

        # add delta to it, run the forward op and then
        # get the new value of the result tensor.
        x_pos = original + delta
        tensor_to_check.set_float_element(i, x_pos)
        y_pos = get_output()

        # Subtract delta from this element, run the op again
        # and get the new value of the result tensor.
        x_neg = original - delta
        tensor_to_check.set_float_element(i, x_neg)
        y_neg = get_output()

        # restore old value
        tensor_to_check.set_float_element(i, original)

        # compute the gradient of this element and store
        # it into a numpy array.
        gradient_flat[i] = (y_pos - y_neg) / delta / 2

    # reshape the gradient result to the shape of the source tensor.
    return gradient_flat.reshape(tensor_to_check.get_dims())
```

## Auto Gradient Check Framework

Each Operator Kernel has three kinds of Gradient:

1. Numerical gradient
2. CPU kernel gradient
3. GPU kernel gradient (if supported by the device)

The numerical gradient only relies on the forward Operator, so we use the numerical gradient as the reference value. The gradient checking is performed in the following three steps:

1. Calculate the numerical gradient
2. Calculate CPU kernel gradient with the backward Operator and compare it with the numerical gradient.
3. Calculate GPU kernel gradient with the backward Operator and compare it with the numeric gradient. (if supported)

#### Python Interface

```python
    def check_grad(self,
                   forward_op,
                   input_vars,
                   inputs_to_check,
                   output_name,
                   no_grad_set=None,
                   only_cpu=False,
                   max_relative_error=0.005):
        """
        :param forward_op: used to create backward_op
        :param input_vars: numpy value of input variable. The following
          computation will use these variables.
        :param inputs_to_check: the input variable with respect to which the
          gradient will be computed.
        :param output_name: The final output variable name.
        :param max_relative_error: The relative tolerance parameter.
        :param no_grad_set: used to create backward ops
        :param only_cpu: only compute and check gradient on cpu kernel.
        :return:
        """
```

### How to check if two numpy arrays are close enough?
if `abs_numerical_grad` is nearly zero, then use absolute error for numerical_grad.

```python
numerical_grad = ...
operator_grad = numpy.array(scope.find_var(grad_var_name(name)).get_tensor())

abs_numerical_grad = numpy.abs(numerical_grad)
# if abs_numerical_grad is nearly zero, then use abs error for
# numeric_grad, instead of relative error.
abs_numerical_grad[abs_numerical_grad < 1e-3] = 1

diff_mat = numpy.abs(abs_numerical_grad - operator_grad) / abs_numerical_grad
max_diff = numpy.max(diff_mat)
```


#### Notes：
The Input data for auto gradient checker should be reasonable to avoid numerical stability problem.


#### References:

- [Gradient checking and advanced optimization(en)](http://deeplearning.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization)
- [Gradient checking and advanced optimization(cn)](http://ufldl.stanford.edu/wiki/index.php/%E6%A2%AF%E5%BA%A6%E6%A3%80%E9%AA%8C%E4%B8%8E%E9%AB%98%E7%BA%A7%E4%BC%98%E5%8C%96)
