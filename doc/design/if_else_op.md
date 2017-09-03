IfOp should have only one branch. An IfOp operator takes a `cond` variable whose value must be a vector of N boolean elements. Its return value has M (M<=N) instances, each corresponds to a true element in `cond`.

```python
import paddle as pd

x = var()
y = var()
cond = var()

b = pd.create_ifop(inputs=[x], output_num=1)
with b.true_block():
    x = b.inputs(0)
    z = operator.add(x, y)
    b.set_output(0, operator.softmax(z))

out = b(cond)
```

If we want the output still has N instances, we can use IfElseOp with a default value, whose minibatch size must be N:

```python
import paddle as pd

x = var()
y = var()
cond = var()
default_value = var()
b = pd.create_ifelseop(inputs=[x], output_num=1)
with b.true_block():
    x = b.inputs(0)
    z = operator.add(x, y)
    b.set_output(0, operator.softmax(z))

with b.false_block():
    x = b.inputs(0)
    z = layer.fc(x)
    b.set_output(0, operator.softmax(z))

out = b(cond)
```

If only true_block is set in an IfElseOp, we can have a default value for false as:
```python
import paddle as pd

x = var()
y = var()
cond = var()
default_value = var()
b = pd.create_ifelseop(inputs=[x], output_num=1, default_value)

with b.true_block():
    x = b.inputs(0)
    z = operator.add(x, y)
    b.set_output(0, operator.softmax(z))

out = b(cond)
```
where default_value is a list of vars for `cond` == False.
