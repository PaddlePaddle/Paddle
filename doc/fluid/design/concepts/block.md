# Design Doc: Block and Scope

## The Representation of Computation

Both deep learning systems and programming languages help users describe computation procedures.  These systems use various representations of computation:

- Caffe, Torch, and Paddle: sequences of layers.
- TensorFlow, Caffe2, Mxnet: graph of operators.
- PaddlePaddle: nested blocks, like C++ and Java programs.

## Block in Programming Languages and Deep Learning

In programming languages, a block is a pair of curly braces that includes local variables definitions and a sequence of instructions or operators.

Blocks work with control flow structures like `if`, `else`, and `for`, which have equivalents in deep learning:

<table>
<thead>
<tr>
<th>programming languages</th>
<th>PaddlePaddle</th>
</tr>
</thead>
<tbody>
<tr>
<td>for, while loop </td>
<td>RNN, WhileOp </td>
</tr>
<tr>
<td>if, if-else, switch </td>
<td>IfElseOp, SwitchOp </td>
</tr>
<tr>
<td>sequential execution </td>
<td>a sequence of layers </td>
</tr>
</tbody>
</table>


A key difference is that a C++ program describes a one pass computation, whereas a deep learning program describes both the forward and backward passes.

## Stack Frames and the Scope Hierarchy

The existence of the backward pass makes the execution of a block of PaddlePaddle different from traditional programs:

<table>
<thead>
<tr>
<th>programming languages</th>
<th>PaddlePaddle</th>
</tr>
</thead>
<tbody>
<tr>
<td>stack </td>
<td>scope hierarchy </td>
</tr>
<tr>
<td>stack frame  </td>
<td>scope </td>
</tr>
<tr>
<td>push at entering block </td>
<td>push at entering block </td>
</tr>
<tr>
<td>pop at leaving block </td>
<td>destroy when minibatch completes </td>
</tr>
</tbody>
</table>


1. In traditional programs:

   - When the execution enters the left curly brace of a block, the runtime pushes a frame into the stack, where it realizes local variables.
   - After the execution leaves the right curly brace, the runtime pops the frame.
   - The maximum number of frames in the stack is the maximum depth of nested blocks.

1. In PaddlePaddle

   - When the execution enters a block, PaddlePaddle adds a new scope, where it realizes variables.
   - PaddlePaddle doesn't pop a scope after the execution of the block because variables therein are used by the backward pass.  So it has a stack forest known as a *scope hierarchy*.
   - The height of the highest tree is the maximum depth of nested blocks.
   - After the processing of a minibatch, PaddlePaddle destroys the scope hierarchy.

## Use Blocks in C++ and PaddlePaddle Programs

Let us consolidate the discussion by presenting some examples.

### Blocks with `if-else` and `IfElseOp`

The following C++ programs shows how blocks are used with the `if-else` structure:

```c++
namespace pd = paddle;

int x = 10;
int y = 1;
int z = 10;
bool cond = false;
int o1, o2;
if (cond) {
  int z = x + y;
  o1 = z;
  o2 = pd::layer::softmax(z);
} else {
  int d = pd::layer::fc(z);
  o1 = d;
  o2 = d+1;
}

```

An equivalent PaddlePaddle program from the design doc of the [IfElseOp operator](../execution/if_else_op.md) is as follows:

```python
import paddle as pd

x = minibatch([10, 20, 30]) # shape=[None, 1]
y = var(1) # shape=[1], value=1
z = minibatch([10, 20, 30]) # shape=[None, 1]
cond = larger_than(x, 15) # [false, true, true]

ie = pd.ifelse()
with ie.true_block():
    d = pd.layer.add_scalar(x, y)
    ie.output(d, pd.layer.softmax(d))
with ie.false_block():
    d = pd.layer.fc(z)
    ie.output(d, d+1)
o1, o2 = ie(cond)
```

In both examples, the left branch computes `x+y` and `softmax(x+y)`, the right branch computes `fc(x)` and `x+1` .

The difference is that variables in the C++ program contain scalar values, whereas those in the PaddlePaddle programs are mini-batches of instances.


### Blocks with `for` and `RNNOp`

The following RNN model in PaddlePaddle from the [RNN design doc](../dynamic_rnn/rnn.md) :

```python
x = sequence([10, 20, 30]) # shape=[None, 1]
m = var(0) # shape=[1]
W = var(0.314, param=true) # shape=[1]
U = var(0.375, param=true) # shape=[1]

rnn = pd.rnn()
with rnn.step():
  h = rnn.memory(init = m)
  h_prev = rnn.previous_memory(h)
  a = layer.fc(W, x)
  b = layer.fc(U, h_prev)  
  s = pd.add(a, b)
  act = pd.sigmoid(s)
  rnn.update_memory(h, act)
  rnn.output(a, b)
o1, o2 = rnn()
```
has its equivalent C++ program as follows

```c++
int* x = {10, 20, 30};
int* m = {0};
int* W = {0.314};
int* U = {0.375};

int mem[sizeof(x) / sizeof(x[0]) + 1];
int o1[sizeof(x) / sizeof(x[0]) + 1];
int o2[sizeof(x) / sizeof(x[0]) + 1];
for (int i = 1; i <= sizeof(x)/sizeof(x[0]); ++i) {
  int x = x[i-1];
  if (i == 1) mem[0] = m;
  int a = W * x;
  int b = Y * mem[i-1];
  int s = fc_out + hidden_out;
  int act = sigmoid(sum);
  mem[i] = act;
  o1[i] = act;
  o2[i] = hidden_out;
}
```

## Compilation and Execution

Like TensorFlow, a PaddlePaddle program is written in Python. The first part describes a neural network as a protobuf message, and the rest executes the message for training or inference.

The generation of this protobuf message is similar to how a compiler generates a binary executable file. The execution of the message is similar to how the OS executes the binary file.

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
  inputs = {0} // the index of x in vars of BlockDesc above
  outputs = {5, 3} // indices of act and hidden_out in vars of BlockDesc above
  attrs {
    "states" : {1} // the index of h
    "step_net" : <above step net>
  }
};
```

This `OpDesc` value is in the `ops` field of the `BlockDesc` value representing the global block.


## The Compilation of Blocks

During the generation of the Protobuf message, the Block should store VarDesc (the Protobuf message which describes Variable) and OpDesc (the Protobuf message which describes Operator).

VarDesc in a block should have its name scope to avoid local variables affecting parent block's name scope.
Child block's name scopes should inherit the parent's so that OpDesc in child block can reference a VarDesc that is stored in the parent block. For example:

```python
a = pd.Variable(shape=[20, 20])
b = pd.fc(a, params=["fc.w", "fc.b"])

rnn = pd.create_rnn()
with rnn.stepnet():
    x = a.as_step_input()
    # reuse fc's parameter
    fc_without_b = pd.get_variable("fc.w")
    rnn.output(fc_without_b)

out = rnn()
```
The method `pd.get_variable` can help retrieve a Variable by the name. The Variable may be stored in a parent block, but might be retrieved in a child block, so block should have a variable scope that supports inheritance.

In compiler design, the symbol table is a data structure created and maintained by compilers to store information about the occurrence of various entities such as variable names, function names, classes, etc.

To store the definition of variables and operators, we define a C++ class `SymbolTable`, like the one used in compilers.

`SymbolTable` can do the following:

- store the definitions (some names and attributes) of variables and operators,
- verify if a variable was declared,
- make it possible to implement type checking (offer Protobuf message pointers to `InferShape` handlers).


```c++
// Information in SymbolTable is enough to trace the dependency graph. So maybe
// the Eval() interface takes a SymbolTable is enough.
class SymbolTable {
 public:
  SymbolTable(SymbolTable* parent) : parent_(parent) {}

  OpDesc* NewOp(const string& name="");

  // TODO determine whether name is generated by python or C++.
  // Currently assume that a unique name will be generated by C++ if the
  // argument name is left default.
  VarDesc* Var(const string& name="");

  // find a VarDesc by name, if recursive is true, find parent's SymbolTable
  // recursively.
  // this interface is introduced to support InferShape, find protobuf messages
  // of variables and operators, pass pointers into InferShape.
  //
  // NOTE maybe some C++ classes such as VarDescBuilder and OpDescBuilder should
  // be proposed and embedded into pybind to enable python operation on C++ pointers.
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

The `Block` class takes a `BlockDesc` as input, and provides `Run` and `InferShape` functions.


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
           const platform::Place& place) const override {
    PADDLE_ENFORCE(symbols_ready_, "operators and variables should be created first.");
    for (auto& op : runtime_table_.ops()) {
      op->Run(scope, place);
    }
  }

  void CreateVariables(const framework::Scope& scope);
  void CreateOperators();

  // some other necessary interfaces of NetOp are listed below
  // ...

private:
  BlockDesc desc_;
  bool symbols_ready_{false};
};
```

## The Execution of Blocks

Block inherits from OperatorBase, which has a Run method.
Block's Run method will run its operators sequentially.

There is another important interface called `Eval`, which takes some arguments called targets and generates a minimal graph which treats targets as the end points and creates a new Block. After `Run`, `Eval` will get the latest value and return the targets.

The definition of Eval is as follows:

```c++
// clean a block description by targets using the corresponding dependency graph.
// return a new BlockDesc with minimal number of operators.
// NOTE: The return type is not a Block but the block's description so that this can be distributed
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
