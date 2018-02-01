### `scalar_switch_case_op` Design

### Background
In a general programing language, there are many scalar control flow operators such as `if` `else` `switch-case`, these operators take scalar bool values as condition input, then decide which block of code the program should run. PaddlePaddle fluid also need such kind of operators.

### For example:
In learning rate decay, one strategy is called `piecewise_decay`, it has two inputs: `boundaries` and `values`.

If the input parameters are:

```
boundaries = [100000, 110000]
values = [1.0, 0.5, 0.1]
```

then the decay logic should be:

```python
def piecewise_decay()
  learning_rate = init_lr

  if step <= 100000:
    learning_rate = 1.0
  elif step > 100000 and step <= 110000:
    learning_rate = 0.5
  else:
    learning_rate = 0.1

  return learning_rate
```

### Solution
To meet this kind of demand, we can have an `scalar_switch_case_op`:

It's interface is:

```python
def piecewise_decay():
  global_step = pd.Var(shape=[1], value=...)
  learning_rate = pd.Var(shape=[1], value=...)

  candition1 = pd.less_equal(global_step, 100000)
  candition2 = pd.and(pd.less_then(100000, global_step), pd.less)

  switch = ScalarSwitchOp()
  with switch.case(candition1):
    pd.assign(learning_rate, 1.0)
  with switch.case(candition2):
    pd.assign(learning_rate, 0.5)
  with switch.default_case():
    pd.assign(learning_rate, 0.1)

  return learning_rate
``` 

For `scalar_switch_case_op` op, the candition var should be a scalar Variable, which means it's shape is `[1]`.

`scalar_switch_case_op` can be easily wrapped as `scalar_if_else` operator

```python
# scalar_if_else

cand = pd.less_equal(global_step, 100000)

if_op = ScalarIfElseOp(cond)
with if_op.true_block()
  ...
with if_op.false_block()
  ...

```
each block above can be ignored so it can be use as `if(cond)` or `if(!cond)`
