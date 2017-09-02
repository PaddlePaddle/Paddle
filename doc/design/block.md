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
  
## Block Implementation
Currentlly, there is a simple implementation of blockin [User Interface Design](), which has the most functions of a Block when user writes a program. 

When compiling user's program, Block will store both `VarDesc` and `OperatorDesc`. During execution, Block will first create all `Variable`s and `Operator`s according the descriptions of `VarDesc`s and `OperatorDesc`s, then executes all the operators.



```c++
#include <map>
#include <memory>

struct VarDesc;
struct BlockDesc;

// VarDescScope is similar to Scope, but offer a simpler map with the same value type.
class VarDescScope {
 public:
  VarDescScope();
  // lookup a variable description, recursively or not.
  VarDesc* Lookup(const std::string& name, bool recursive=true);
  // create a new description in local namescope.
  VarDesc* New(const std::string& name);
  // delete a local variable's description
  void Delete(const std::string& name);
 private:
  std::map<std::string, VarDesc> map_;
  VarDescScope* father_{nullptr};
};

/*
 * Block represents a group of operators.
 */
class Block : public OperatorBase {
 public:
  // Block which will create its own scope when run
  Block(Block* father) : father_block_(father) {}

  // Block with a given scope
  Block(Block* father, Scope* scope) : father_block_(father), scope_(scope) {}

  // For RNN's step scopes.
  void SetScope(Scope* scope);

  // NOTE Block has its own scope, so the argument scope here is not needed.
  virtual void Run(const Scope& scope,
                   const platform::DeviceContext& dev_ctx) const;

  // Evaluate some specific variables.
  void Eval(std::vector<std::string> vars);

  // TODO most interfaces of NetOp

  // initial scope, create if scope_ is empty.
  void InitScope();

  BlockDesc Serialize();
  void Deserialize(const BlockDesc& desc);

 private:
  Block* father_block_;
  // All the VarDescs are store in var_desc_lib_;
  std::unique_ptr<VarDescScope> var_desc_lib_;

  Scope* scope_;
  bool scope_owner_{false};

  std::vector<OperatorBase> ops_;
};
```

python wrapper

```python
import paddle as pd
import paddle.v2.framework.core as core

class Block(pd.Op):
    '''
    Block is a group of operators.
    '''
    def __init__(self):
        self._block = core.Block()

    def append(cmd):
        '''
        a cmd can be a op or a soft block which is implemented by python.
        '''
        if isinstance(cmd, Op):
            self._block.AppendOp(Op.core_op)
        else:
            pass

    def run(self, device=None):
        '''
        device: device context
        '''
        self.scope = core.Scope()
        self._block.run(self.scope, device)

    def eval(self, targets):
        '''
        targets: list of Variable
        '''
        var_names = [v.name for v in targets]
        self._block.eval(var_names)
        return targets

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass

```


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
