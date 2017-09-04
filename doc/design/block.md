# Design Doc: Use Block in RNNOp, While Op, IfElseOp

In C++ and Java programming language, a block is a lexical structure of source code which is grouped as one line of code.

RNNOp looks like the loop structure in programming languages.
And similarly, WhileOp and IfElseOp are like loop and conditions respectively.
So we want to verify if we should have a class Block in PaddlePaddle that works like a pair of curly braces in the loop and condition structures of programming languages.

Blocks do not only group source code, but also narrow the lexical scope of variables so that they do not conflict with variables having the same name used elsewhere in a program.

In Paddle, we need a similar concept called Block to support following scenes:

- define a PaddlePaddle program by writing blocks of codes, which includes the definitions of variables and operators.
  - `RNNOp`, `SwitchOp`, `WhileOp` and `IfElseOp` needs Block to help to define sub-block. 
- help to execute multiple operators, blocks should group operators and runs like a single operator.

## How to use Block
In `RNNOp`, `SwitchOp`, `WhileOp` and `IfElseOp`, a sub-block should be used to help to define a sub-block.

Let's start from how a RNNOp is described using Block:

```python
v = some_op()
m_boot = some_op()

W = pd.Variable(shape=[20, 20])
U = pd.Varable(shape=[20, 20])

rnn = RNNOp()
with rnn.stepnet() as net:
  x = net.add_input(v)
  h = net.add_memory(init=m_boot)
  
  fc_out = pd.matmul(W, x)
  hidden_out = pd.matmul(U, h)
  sum = pd.add_two(fc_out, hidden_out)
  act = pd.sigmoid(sum)
  # declare outputs
  net.add_output(act, hidden_out)

acts, hs = rnn()
```

This python program will be transformed into Protobuf messages which describe the model,  passes it to a C++ framework and creates the corresponding Variables and Operators, and execute all the operators.

## Block Implementation

During the Protobuf message generation, the Block should store VarDesc (the Protobuf message which describes Variable) and OpDesc (the Protobuf message which describes Operator).

VarDesc in a block should have its name scope to avoid local variables affect father block's name scope. 
Child block's name scopes should inherit the father's so that OpDesc in child block can reference a VarDesc that stored in father block. For example

```python
a = pd.Varaible(shape=[20, 20])
b = pd.fc(a, params=["fc.w", "fc.b"])

rnn = pd.create_RNNOp()
with rnn.stepnet() as net:
    x = net.add_input(a)
    # reuse fc's parameter
    fc_without_b = pd.get_variable("fc.w")
    net.add_output(fc_without_b)

out = rnn()
```
the method `pd.get_variable` can help retrieve a Variable by a name, a Variable may store in a father block, but might be retrieved in a child block, so block should have a variable scope that supports inheritance.

We can implement this idea in the following approach

```c++
class Block {
public:
  Block(Block* father) : father_block_{father} {}
  // add a new VarDesc, return the pointer to enable other functions.
  // NOTE Will check whether some variable called the same name.
  VarDesc* AddVarDesc();
  OpDesc* AddOpDesc();

  // name: variable's name
  // recursive: whether to find the variable in father's scope recursively.
  VarDesc* FindVarDesc(const string& name, bool recursive=true);

private:
  Block* father_block_;
  // descriptions
  map<VarDesc> var_descs_;
  map<OpDesc> op_descs_;
}
```

after all the VarDescs and OpDescs are added into Block, the block is ready to run.
During the running of Block, it first creates all Variables according to VarDescs and Operators according to OpDescs.

What's more, each block will have its scope.

The codes are as follows:

```c++
class Block : public OperatorBase {
public:
  // ...
  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const {
    RuntimeInit();
    LocalScopeInit(scope);

    for (auto* op : ops_) {
      op->Run(scope, dev_ctx);
    }
  }

  void LocalScopeInit(const framework::Scope &scope) {
    if (!scope_) {
      scope_ = Scope.NewScope();
    }
  }

protected:
  // create all variables and operators according to Protobuf description.
  RuntimeInit() {
    if (ops_.empty()) {
      // create ops
      // create vars
    }
  }

private:
  // ...
  vector<Operator*> ops_;
  vector<Variable*> vars_;
  // local scope
  Scope* scope_;
};
```
## Run and Eval targets
Block inherits from OperatorBase, which has a Run method. 
Block's Run method will run its operators sequentially.

There is another important interface called Eval, which passed in some variables called targets, and Eval will generate a minimal graph which takes targets as the end points and creates a new Block, 
after Run, Eval will get the latest value and return the targets.

The definition of Eval is as follows:

```c++
class Block : public OperatorBase {
public:
  // ...
  vector<Variable*> Eval(const vector<string>& targets) {
    // 1. generate a minimal graph which takes targets as end points

    // 2. extract all operators in this graph and generate a new block which
    // take this as the father block

    // 3. return all the variables of the targets
  }
  // ...
}
```

In Python binding, there is a default Block called `g_block` which is hidden from users.
Most of time, user should not feel the existence of Block, so a function is provided as `pd.eval([targets])`,
let's take a example:

```python
import paddle as pd
a = pd.Variable(shape=[20, 30])
b = pd.Variable(shape=[20, 30])

c = pd.fc(a)
d = pd.fc(c)

c_val, d_val = pd.eval([c, d])
```
