# Optimizer Design

## The Problem

A PaddlePaddle program, or a block, is a sequence of operators operating variables.  A training program needs to do three kinds of works:

1. the forward pass, which computes intermediate results and the cost(s),
1. the backward pass, which derives gradients from intermediate results and costs, and
1. the optimization pass, which update model parameters to optimize the cost(s).

These works rely on three kinds of operators:

1. forward operators,
1. gradient operators, and
1. optimization operators.

It's true that users should be able to create all these operators manually by calling some low-level API, but it would be much more convenient if they could only describe the forward pass and let PaddlePaddle create the backward and optimization operators automatically.

In this design, we propose a high-level API that automatically derives the optimisation pass and operators from the forward pass.


## High-level Python API to describe the training process

1. User write code to describe the network:

	```python
	images = layer.data("images")
	labels = layer.data("labels")
	w1 = pd.var("w1")
	b1 = pd.var("b1")
	hidden = layer.fc(images, w=w1, b=b1)
	cost = layer.mse(hidden, labels)
	```

	The above code snippet will create forward operators in [Block](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/block.md).


2. Users create a certain kind of Optimizer with some argument.

	```python
	optimizer = AdagradOptimizer(learing_rate=0.001)
	```

3. Users use the optimizer to `minimize` a certain `cost` through updating parameters in parameter_list.

	```python
	opt_op_list = optimizer.minimize(cost, parameter_list=[w1, b1])
	```
	The above code snippet will create gradient and optimization operators in Block. The return value of `minimize()` is list of optimization operators that will be run by session.

4. Users use Session/Executor to run this opt_op_list as target to do training.

	```python
	sess.run(target= opt_op_list, ...)
	```

### Optimizer Python interface:

```python
class Optimizer(object):
    """Optimizer Base class.

    """

    def __init__(self):
        pass

    def create_optimization_pass(self, parameters_and_grads):
        """Add optimization operators to update gradients to variables.

        Args:
          parameters_and_grads: a list of (variable, gradient) pair to update.

        Returns:
          optmization_op_list: a list of optimization operator that will update parameter using gradient.
        """
        return None

    def minimize(self, loss, parameter_list):
        """Add operations to minimize `loss` by updating `parameter_list`.

        This method combines interface `append_backward()` and
        `create_optimization_pass()` into one.
        """
        params_grads = self.create_backward_pass(loss, parameter_list)
        update_ops = self.create_optimization_pass(params_grads)
        return update_ops

```

Users can inherit the Optimizer above to create their own Optimizer with some special logic, such as AdagradOptimizer.
