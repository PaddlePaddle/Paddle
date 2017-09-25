## Auto Gradient Checker Design

## Backgraound：
- Generally, it is easy to check whether the forward computation of an Operator is correct or not. However, backpropagation is a notoriously difficult algorithm to debug and get right:
  1. you should get the right backpropagation formula according to the forward computation.
  2. you should implement it right in CPP.
  3. it's difficult to prepare test data.

- Auto gradient checking gets a numerical gradient by forward Operator and use it as a reference of the backward Operator's result. It has several advantages:
  1. numerical gradient checker only need forward operator.
  2. user only need to prepare the input data for forward Operator.

## Mathematical Theory
The following two document from Stanford has a detailed explanation of how to get numerical gradient and why it's useful.

- [Gradient checking and advanced optimization(en)](http://deeplearning.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization)
- [Gradient checking and advanced optimization(cn)](http://ufldl.stanford.edu/wiki/index.php/%E6%A2%AF%E5%BA%A6%E6%A3%80%E9%AA%8C%E4%B8%8E%E9%AB%98%E7%BA%A7%E4%BC%98%E5%8C%96)


## Numeric Gradient Implementation
### Python Interface
```python
def get_numerical_gradient(op,
                         input_values,
                         output_name,
                         input_to_check,
                         delta=0.005,
                         local_scope=None):
    """
    Get Numeric Gradient for an operator's input.

    :param op: C++ operator instance, could be an network
    :param input_values: The input variables. Should be an dictionary, whose key is
    variable name, and value is numpy array.
    :param output_name: The final output variable name.
    :param input_to_check: The input variable with respect to which to compute the gradient.
    :param delta: The perturbation value for numeric gradient method. The
    smaller delta is, the more accurate result will get. But if that delta is
     too small, it will suffer from numerical stability problem.
    :param local_scope: The local scope used for get_numeric_gradient.
    :return: The gradient array in numpy format.
    """
```

### Explaination:

- Why need `output_name`
  - An Operator may have multiple Output, one can get independent gradient from each Output. So caller should specify the name of the output variable.

- Why need `input_to_check`
  - One operator may have multiple inputs. Gradient Op can calculate the gradient of these inputs at the same time. But Numeric Gradient needs to calculate them one by one. So `get_numeric_gradient` is designed to calculate the gradient for one input. If you need to compute multiple inputs, you can call `get_numeric_gradient` multiple times.


### Core Algorithm Implementation


```python
    # we only compute gradient of one element a time.
    # we use a for loop to compute the gradient of each element.
    for i in xrange(tensor_size):
        # get one input element by its index i.
        origin = tensor_to_check.get_float_element(i)

        # add delta to it, run op and then get the new value of the result tensor.
        x_pos = origin + delta
        tensor_to_check.set_float_element(i, x_pos)
        y_pos = get_output()

        # plus delta to this element, run op and get the new value of the result tensor.
        x_neg = origin - delta
        tensor_to_check.set_float_element(i, x_neg)
        y_neg = get_output()

        # restore old value
        tensor_to_check.set_float_element(i, origin)

        # compute the gradient of this element and store it into a numpy array.
        gradient_flat[i] = (y_pos - y_neg) / delta / 2

    # reshape the gradient result to the shape of the source tensor.
    return gradient_flat.reshape(tensor_to_check.get_dims())
```

## Auto Graident Checker Framework

Each Operator Kernel has three kinds of Gradient:

1. Numerical gradient
2. CPU kernel gradient
3. GPU kernel gradient (if supported)

The numerical gradient only relies on forward Operator. So we use the numerical gradient as the reference value. And the gradient checking is performed in the following three steps:

1. calculate the numerical gradient
2. calculate CPU kernel gradient with the backward Operator and compare it with the numerical gradient
3. calculate GPU kernel gradient with the backward Operator and compare it with the numeric gradient (if supported)

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
        :param inputs_to_check: the input variable with respect to which to compute the gradient.
        :param output_name: The final output variable name.
        :param max_relative_error: The relative tolerance parameter.
        :param no_grad_set: used when create backward ops
        :param only_cpu: only compute and check gradient on cpu kernel.
        :return:
        """
```

### How to check if two numpy array is close enough?
if `abs_numerical_grad` is nearly zero, then use abs error for numerical_grad

```python
numerical_grad = ...
operator_grad = numpy.array(scope.find_var(grad_var_name(name)).get_tensor())

abs_numerical_grad = numpy.abs(numerical_grad)
# if abs_numerical_grad is nearly zero, then use abs error for numeric_grad, not relative
# error.
abs_numerical_grad[abs_numerical_grad < 1e-3] = 1

diff_mat = numpy.abs(abs_numerical_grad - operator_grad) / abs_numerical_grad
max_diff = numpy.max(diff_mat)
```


#### Notes：
The Input data for auto gradient checker should be reasonable to avoid numerical  stability problem.


#### Refs:

- [Gradient checking and advanced optimization(en)](http://deeplearning.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization)
- [Gradient checking and advanced optimization(cn)](http://ufldl.stanford.edu/wiki/index.php/%E6%A2%AF%E5%BA%A6%E6%A3%80%E9%AA%8C%E4%B8%8E%E9%AB%98%E7%BA%A7%E4%BC%98%E5%8C%96)
