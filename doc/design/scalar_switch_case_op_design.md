### Scalar `switch_case_op` Design

### Background
In a general programing language, there are many scalar control flow operators such as `if` `else` `switch-case`, these operators take scalar bool values as condition input, then decide which block of code the program should run. PaddlePaddle fluid also need such kind of operators.

### For example:
In learning rate decay, one strategy is called `piecewise_decay`, it has two inputs: `boundaries` and `values`

```
boundaries = [100000, 110000]
values = [1.0, 0.5, 0.1]
```

the decay logic should be:

```python
if step <= 100000:
  return 1.0
elif step > 100000 and step <= 110000:
  return 0.5
else:
  return 0.1
```

### Solution
To meet this kind of demand, we can have an `scalar_switch_case_op`:

It's interface is:

```python
global_step = pd.Var(shape=[1], value=...)
learning_rate = pd.Var(shape=[1], value=...)

candition1 = pd.less_equal(global_step, 100000)
candition2 = pd.and(pd.less_then(100000, global_step), pd.less)

switch = SwitchOp()
with switch.case(candition1):
	pd.assign(learning_rate, 1.0)
with switch.case(candition2):
	pd.assign(learning_rate, 0.5)
with switch.default_case():
   pd.assign(learning_rate, 0.1)
``` 

For `scalar_switch_case_op` op, the candition var should be a scalar Variable, which means it's shape is `[1]`.
