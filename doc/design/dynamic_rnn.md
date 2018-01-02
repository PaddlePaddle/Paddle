# Implementation Doc: Dynamic RNN

## A glance of Dynamic RNN

A common neural network structure called recurrent neural network(`RNN` for short), which there is a directed circle in the neural network model. RNN can use a internal memory to process arbitrary sequences of inputs.

PaddlePaddle Fluid directly represents the `directed circle` in the `ProgramDesc`, since we do not use directed acyclic graph to represent our model. The `ProgramDesc` just like the AST of a programming language, which describes the computation instructions for training a neural network. We use arrays and a while loop to describe the training process of an RNN. The C++ code below demonstrates the forward logic of RNN which PaddlePaddle Fluid generates in `ProgramDesc`.

```cpp
auto input = LoDTensor(...);  // LoDTensor is the data structure for time series

std::vector<LoDTensor> inputs_for_each_timestep = LoDTensorToTimesteps(LoDTensor())
std::vector<LoDTensor> memories;
memories.resize(inputs_for_each_timestep.size() + 1);
memories[0] = 0;
std::vector<LoDTensor> outputs_for_each_timestep;
outputs_for_each_timestep.resize(inputs_for_each_timestep.size());

auto W0 = LoDTensor(...);
auto W1 = LoDTensor(...);
auto Bias = LoDTensor(...);

size_t i = 0;
while (i < inputs_for_each_timestep.size()) {
  auto& step_input = inputs_for_each_timestep[i];
  auto& ex_mem = memories[i];
  
  auto tmp0 = step_input * W0;
  auto tmp1 = ex_mem * W1;
  auto sum = tmp0 + tmp1 + Bias
  auto hidden = sigmoid(sum);
  memories[i+1] = sum;
  outputs_for_each_timestep[i] = sum;
}

LoDTensor outputs = TimestepsToLoDTensor(outputs_for_each_timestep);
```

The `Dynamic RNN` in PaddlePaddle Fluid is basically a syntax sugar to compose operators, such as `while`, `split_lod_tensor_to_timesteps`, `restore_lod_tensor_from_timesteps`.

The following of this document will be organized in several sections:

1. Control flow operators
1. Data manipulation operators of RNN.
2. Backward of RNN.


## Control flow operators

### WhileOp

The primary control flow operator to implement dynamic RNN is `WhileOp`. The `WhileOp` takes a sub-block. The operators in the sub-block will be executed again and again while the condition is true. The fragment program of a while op and its sub-block is:

```text
program {
  block {
    idx: 0  # main block
    parent_idx: -1  # -1 means no parent
    ops: {
      ...  # ops before while op
      
      op {
        inputs: ...,
        outputs: ...,
        type: "while",
        attrs: {
          attr {
            name: 'sub_block',
            type: 'BlockID',
            value: 1  # the sub block id of this while op is 1
          }
        }
      }
      ...  # ops after while op
    }
  }
  
  block {
    idx: 1  # the sub_block of while_op
    parent_idx: 0  # parent of while block is the main block
    ops: {
      ... # ops inside while
    }
  }
}
```

The while 
