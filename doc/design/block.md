# Block design

In computer programming, a block is a lexical structure of source code which is grouped together.
In most programming languages, block is useful when define a function or some conditional statements such as `if` and `while`.

In Paddle, we need a similar concept to support following scenes:

- the basic unit of execution, just like the original `NetOp`, block can group multiple operators and make them behave just like a single operator. Unlike `NetOp`, a network should not be an operator, but a block can be treated as a operator.
  - Block should have a `Exec` method, which will execute the operators grouped in the block.
- store the context for operators, including the information of variables they will operator on. 
- Support the definition and execution of special operators such as `RNNOp`, `switch_op`, `while_op`, `if_else_op`, which needs one or more code-blocks.
  - Block will help to define and execute thouse operators.
  
## when need a block
Blocks are needed in following code snippets
```c++
if(cond) { block_true } else { block_false }
```

```c++
while(cond){ block }
```

```c++
int func(inputs) { block with outputs }
```

```c++
namespace xxxx {
  block
}
```

Paddle will have `if_else_op`, `switch_op`, `while_op` and some other operators that need a function block such as RNNOp, and enable definition and execution of a block is vital to those operators.

What's more, Paddle should has a `pd.namespace`, which will help to preventing name conflicts of variables in large model.

For example:

```python
import paddle as pd

def submodel(namespace, x):
    # begin a namespace
    with pd.namespace(namespace):
        # should be complex network, here easy one for demo
        # variable W's name is namespace + "W"
        # so in different namespace, there is a W without conflicts
        W = pd.get_variable("W", shape=[30, 30], reuse=True)
        y = pd.matmul(W, x)
        return y

inputs = [a, b, c]

# create 3 submodels and each one has its own parameters.
a_out = submodel("model_a", a)
b_out = submodel("model_b", b)
c_out = submodel("model_c", c)


# d is some op's output
d = someop()

# reuse model_a
d_model_a_out = submodel("model_a", d)
```

with `pd.namespace`, user create 3 different namespaces and in each one, the parameters those have specific name will be reused.

## Why use Block to replace NetOp
- It is weird to treat Network as an operator, but Block in programming language means grouping multiple operators and acts like a single operator, treat a Block as an operator seems more reasonable.
- The basic concepts including`Variable`, `Operator` and `Block` are concepts in programming language, but `NetOp` is not.
- Block is a concept in programming language, it behaves similar to current NetOp, but much concrete, and can guide our futher design.
  - a block looks like something with a pair of curly bracket `{}`, in python we can use `with` statement to make a dramatic code block.
  - a block will contains its operators(just like NetOp) and some local variables
  - operator inside a block can make operation on global variables
  
## Implementation
Currentlly, there is a simple implementation in [User Interface Design](), which has the most function of a Block when user writes a program. 
But it relays on `NetOp` when executes and that works.

We may rename `NetOp` in cpp code to `Block`, but whether to add more functions of a real block into that implementation depends on whether we want to have a `VarDesc` and split the compilation and execution period.

In my opition, Block is very basic concept in underlying implementation, so a python wrapper and a cpp minimal implementation is enough now.
It is free to change if other module needs more complex supports such as `VarDesc` is added.

## `with` statement
Currently Block concept is invisible to users, but some usage are embedded in python interface.

```python
op = pd.ifelseop()

with op.true_block():
    # declare inputs
    a = op.add_input(x)
    # some complex net
    b = pd.fc(a)
    # declare output
    op.add_output(b)
    
with op.false_block():
    # declare inputs
    a = op.add_input(x)
    # some complex net
    b = pd.fc(a)
    # declare output
    op.add_output(b)
```

```python
rnn = pd.rnn_op()

with rnn.stepnet():
    # declare inputs
    a = op.add_input(x)
    b = op.add_input(y)
    # some complex net
    c = pd.fc(a)
    # declare output
    op.add_output(c)
```

```python
op = pd.while_op()
with op.stepnet():
    # do not need to declare inputs or outputs
    c = pd.fc(a)
```
