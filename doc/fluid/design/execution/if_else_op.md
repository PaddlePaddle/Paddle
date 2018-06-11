# The `IfElse` Operator

PaddlePaddle's `IfElse` operator differs from TensorFlow's:

- the TensorFlow version takes a scalar boolean value as the condition so that the whole mini-batch goes to either the true or the false branch, whereas
- the PaddlePaddle version takes a vector of boolean value as the condition, and instances corresponding to true values go to the true branch, those corresponding to false values go to the false branch.

## Example

The following PaddlePaddle program shows the usage of the IfElse operator:

```python
import paddle as pd

x = minibatch([10, 20, 30]) # shape=[None, 1]
y = var(1) # shape=[1], value=1
z = minibatch([10, 20, 30]) # shape=[None, 1]
cond = larger_than(x, 15) # [false, true, true]

ie = pd.ifelse()
with ie.true_block():
    d = pd.layer.add(x, y)
    ie.output(d, pd.layer.softmax(d))
with ie.false_block():
    d = pd.layer.fc(z)
    ie.output(d, d+1)
o1, o2 = ie(cond)
```

A challenge to implement the `IfElse` operator is to infer those variables to be split, or, say, to identify the variable of the mini-batch or those derived from the mini-batch.

An equivalent C++ program is as follows:

```c++
namespace pd = paddle;

int x = 10;
int y = 1;
int z = 10;
bool cond = false;
int o1, o2;
if (cond) {
  int d = x + y;
  o1 = z;
  o2 = pd::layer::softmax(z);
} else {
  int d = pd::layer::fc(z);
  o1 = d;
  o2 = d+1;
}
```
