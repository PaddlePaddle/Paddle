# Design Doc: Block and Scope

## The Representation of Computation

Both deep learning systems and programming languages help users describe computation procedures.  These systems use various representations of computation:

- Caffe, Torch, and Paddle: sequences of layers.
- TensorFlow, Caffe2, Mxnet: graphs of operators.
- PaddlePaddle: nested blocks, like C++ and Java programs.

## Block in Programming Languages and Deep Learning

In programming languages, a block is a pair of curly braces that includes local variables definitions and a sequence of instructions, or operators.

Blocks work with control flow structures like `if`, `else`, and `for`, which have equivalents in deep learning:

| programming languages | PaddlePaddle          |
|-----------------------|-----------------------|
| for, while loop       | RNN, WhileOp          |
| if, if-else, switch   | IfElseOp, SwitchOp    |
| sequential execution  | a sequence of layers  |

A key difference is that a C++ program describes a one pass computation, whereas a deep learning program describes both the forward and backward passes.

## Stack Frames and the Scope Hierarchy

The existence of the backward makes the execution of a block of traditional programs and PaddlePaddle different to each other:

| programming languages | PaddlePaddle                  |
|-----------------------|-------------------------------|
| stack                 | scope hierarchy               |
| stack frame           | scope                         |
| push at entering block| push at entering block        |
| pop at leaving block  | destroy at minibatch completes|

1. In traditional programs:

   - When the execution enters the left curly brace of a block, the runtime pushes a frame into the stack, where it realizes local variables.
   - After the execution leaves the right curly brace, the runtime pops the frame.
   - The maximum number of frames in the stack is the maximum depth of nested blocks.

1. In PaddlePaddle

   - When the execution enters a block, PaddlePaddle adds a new scope, where it realizes variables.
   - PaddlePaddle doesn't pop a scope after the execution of the block because variables therein are to be used by the backward pass.  So it has a stack forest known as a *scope hierarchy*.
   - The height of the highest tree is the maximum depth of nested blocks.
   - After the process of a minibatch, PaddlePaddle destroys the scope hierarchy.

## Use Blocks in C++ and PaddlePaddle Programs

Let us consolidate the discussion by presenting some examples.

### Blocks with `if-else` and `IfElseOp`

The following C++ programs shows how blocks are used with the `if-else` structure:

```c++
int x = 10;
int y = 20;
int out;
bool cond = false;
if (cond) {
  int z = x + y;
  out = softmax(z);
} else {
  int z = fc(x);
  out = z;
}
```

An equivalent PaddlePaddle program from the design doc of the [IfElseOp operator](./if_else_op.md) is as follows:

```python
import paddle as pd

x = var(10)
y = var(20)
cond = var(false)
ie = pd.create_ifelseop(inputs=[x], output_num=1)
with ie.true_block():
    x = ie.inputs(true, 0)
    z = operator.add(x, y)
    ie.set_output(true, 0, operator.softmax(z))
with ie.false_block():
    x = ie.inputs(false, 0)
    z = layer.fc(x)
    ie.set_output(true, 0, operator.softmax(z))
out = b(cond)
```

In both examples, the left branch computes `softmax(x+y)` and the right branch computes `fc(x)`.

A difference is that variables in the C++ program contain scalar values, whereas those in the PaddlePaddle programs are mini-batches of instances.  The `ie.input(true, 0)` invocation returns instances in the 0-th input, `x`, that corresponds to true values in `cond` as the local variable `x`, where `ie.input(false, 0)` returns instances corresponding to false values.

### Blocks with `for` and `RNNOp`

The following RNN model from the [RNN design doc](./rnn.md)

```python
x = sequence([10, 20, 30])
m = var(0)
W = tensor()
U = tensor()

rnn = create_rnn(inputs=[input])
with rnn.stepnet() as net:
  x = net.set_inputs(0)
  h = net.add_memory(init=m)
  fc_out = pd.matmul(W, x)
  hidden_out = pd.matmul(U, h.pre(n=1))
  sum = pd.add_two(fc_out, hidden_out)
  act = pd.sigmoid(sum)
  h.update(act)                       # update memory with act
  net.set_outputs(0, act, hidden_out) # two outputs

o1, o2 = rnn()
print o1, o2
```

has its equivalent C++ program as follows

```c++
int* x = {10, 20, 30};
int m = 0;
int W = some_value();
int U = some_other_value();

int mem[sizeof(x) / sizeof(x[0]) + 1];
int o1[sizeof(x) / sizeof(x[0]) + 1];
int o2[sizeof(x) / sizeof(x[0]) + 1];
for (int i = 1; i <= sizeof(x)/sizeof(x[0]); ++i) {
  int x = x[i-1];
  if (i == 1) mem[0] = m;
  int fc_out = W * x;
  int hidden_out = Y * mem[i-1];
  int sum = fc_out + hidden_out;
  int act = sigmoid(sum);
  mem[i] = act;
  o1[i] = act;
  o2[i] = hidden_out;
}

print_array(o1);
print_array(o2);
```


## Compilation and Execution

Like TensorFlow programs, a PaddlePaddle program is written in Python.  The first part describes a neural network as a protobuf message, and the rest part executes the message for training or inference.

The generation of this protobuf message is like what a compiler generates a binary executable file.  The execution of the message that the OS executes the binary file.

## The "Binary Executable File Format"

The definition of the protobuf message is as follows:

```protobuf
message BlockDesc {
  repeated VarDesc vars = 1;
  repeated OpDesc ops = 2;
}
```

The step net in above RNN example would look like

```
BlockDesc {
  vars = {
    VarDesc {...} // x
    VarDesc {...} // h
    VarDesc {...} // fc_out
    VarDesc {...} // hidden_out
    VarDesc {...} // sum
    VarDesc {...} // act
  }
  ops = {
    OpDesc {...} // matmul
    OpDesc {...} // add_two
    OpDesc {...} // sigmoid
  }
};
```

Also, the RNN operator in above example is serialized into a protobuf message of type `OpDesc` and would look like:

```
OpDesc {
  inputs = {0} // the index of x
  outputs = {5, 3} // indices of act and hidden_out
  attrs {
    "memories" : {1} // the index of h
    "step_net" : <above step net>
  }
};
```

This `OpDesc` value is in the `ops` field of the `BlockDesc` value representing the global block.


## The Compilation of Blocks

During the generation of the Protobuf message, the Block should store VarDesc (the Protobuf message which describes Variable) and OpDesc (the Protobuf message which describes Operator).

VarDesc in a block should have its name scope to avoid local variables affect parent block's name scope.
Child block's name scopes should inherit the parent's so that OpDesc in child block can reference a VarDesc that stored in parent block. For example

```python
a = pd.Varaible(shape=[20, 20])
b = pd.fc(a, params=["fc.w", "fc.b"])

rnn = pd.create_rnn()
with rnn.stepnet() as net:
    x = net.set_inputs(a)
    # reuse fc's parameter
    fc_without_b = pd.get_variable("fc.w")
    net.set_outputs(fc_without_b)

out = rnn()
```
the method `pd.get_variable` can help retrieve a Variable by a name, a Variable may store in a parent block, but might be retrieved in a child block, so block should have a variable scope that supports inheritance.

In compiler design, the symbol table is a data structure created and maintained by compilers to store information about the occurrence of various entities such as variable names, function names, classes, etc.

To store the definition of variables and operators, we define a C++ class `SymbolTable`, like the one used in compilers.

`SymbolTable` can do the following stuff:

- store the definitions (some names and attributes) of variables and operators,
- to verify if a variable was declared,
- to make it possible to implement type checking (offer Protobuf message pointers to `InferShape` handlers).


```c++
// Information in SymbolTable is enough to trace the dependency graph. So maybe
// the Eval() interface takes a SymbolTable is enough.
class SymbolTable {
 public:
  SymbolTable(SymbolTable* parent) : parent_(parent) {}

  OpDesc* NewOp(const string& name="");

  // TODO determine whether name is generated by python or C++
  // currently assume that a unique name will be generated by C++ if the
  // argument name left default.
  VarDesc* NewVar(const string& name="");

  // find a VarDesc by name, if recursive true, find parent's SymbolTable
  // recursively.
  // this interface is introduced to support InferShape, find protobuf messages
  // of variables and operators, pass pointers into InferShape.
  // operator
  //
  // NOTE maybe some C++ classes such as VarDescBuilder and OpDescBuilder should
  // be proposed and embedded into pybind to enable python operate on C++ pointers.
  VarDesc* FindVar(const string& name, bool recursive=true);

  OpDesc* FindOp(const string& name);

  BlockDesc Compile() const;

 private:
  SymbolTable* parent_;

  map<string, OpDesc> ops_;
  map<string, VarDesc> vars_;
};
```

After all the description of variables and operators is added into SymbolTable,
the block has enough information to run.

The `Block` class takes a `BlockDesc` as input, and provide `Run` and `InferShape` functions.


```c++
namespace {

class Block : OperatorBase {
public:
  Block(const BlockDesc& desc) desc_(desc) {}

  void InferShape(const framework::Scope& scope) const override {
    if (!symbols_ready_) {
      CreateVariables(scope);
      CreateOperators();
    }
    // should run InferShape first.
    for (auto& op : runtime_table_.ops()) {
      op->InferShape(scope);
    }
  }

  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {
    PADDLE_ENFORCE(symbols_ready_, "operators and variables should be created first.");
    for (auto& op : runtime_table_.ops()) {
      op->Run(scope, dev_ctx);
    }
  }

  void CreateVariables(const framework::Scope& scope);
  void CreateOperators();

  // some other necessary interfaces of NetOp are list below
  // ...

private:
  BlockDesc desc_;
  bool symbols_ready_{false};
};
```

## The Execution of Blocks

Block inherits from OperatorBase, which has a Run method.
Block's Run method will run its operators sequentially.

There is another important interface called `Eval`, which take some arguments called targets, and generate a minimal graph which takes targets as the end points and creates a new Block,
after `Run`, `Eval` will get the latest value and return the targets.

The definition of Eval is as follows:

```c++
// clean a block description by targets using the corresponding dependency graph.
// return a new BlockDesc with minimal number of operators.
// NOTE not return a Block but the block's description so that this can be distributed
// to a cluster.
BlockDesc Prune(const BlockDesc& desc, vector<string> targets);

void Block::Eval(const vector<string>& targets,
                 const framework::Scope& scope,
                 const platform::DeviceContext& dev_ctx) {
  BlockDesc min_desc = Prune(desc_, targets);
  Block min_block(min_desc);
  min_block.Run(scope, dev_ctx);
}
```
