## Optimizer Design
In deeplearning system, `Optimizer` is used to optimize(minimize) loss thow updating a list of parameters. 

### A typical training process:

1. run forward to calculate activation using data and parameter.
1. run backward to calculate the gradient of activation and parameter using cost, activation, and parameter.
1. run optimize operators to apply/update the gradient to the corresponding parameter.

### Python Interface to describe the training process

1.
User write code to describe the network:

```python
images = layer.data("images")
labels = layer.data("labels")
w1 = pd.var("w1")
hidden = layer.fc(images, W=w1)
cost = layer.mse(hidden, labels)
```

the code above will generate forward operators in [block](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/block.md).


2.
User create a Optimizer and set parameter list that it need to update.

```python
optimizer = AdagradOptimizer(learing_rate=0.001)
```

3.
User use the optimizer to `minimize` a certain `cost` thow updating parameters in parameter_list.

```python
opt = optimizer.minimize(cost, parameter_list=[w1, ...])
```

The return value of `minimize()` is an Operator that rely on all the optimize operator.

4.
Use Session/Executor to run this opt as target.

```python
sess.run(target=[opt], ...)
```

### What does optimizer do:

In PaddlePaddle, we use block of operators to describe computation. From the Python Interface we described above, we can see that `Optimizer` should add some operators to the computation block:

1. Gradient Ops. Used to calculate the gradients.
2. Optimize Ops. Used to apply gradient to parameters.

#### Optimizer Python interface:

```python
class Optimizer(object):
	def _backward(loss):
		"""
		Add Operators to Compute gradients of `loss` 
		It returns the variables that will be updated for this loss.
		"""
		...
		return variables

	def _update(var_list):
		"""
		Add Operators to Apply gradients to variables 
		in var_list. It returns an update `Operator`.
		Run this operator will trace back to all update and backward
		op related.
		"""
		...
		return update_op

	def minimize(loss, var_list):
		"""Add operations to minimize `loss` by updating `var_list`.
		
		This method simply combines calls `_backward()` and
		`_update()`.
		"""
		variables = _backward(loss)
		update_op = _update(variables)
		return update_op
```

because we do not want users to know the step of `_backward` and `_update`, so we decide to export only `minimize()` to users.
