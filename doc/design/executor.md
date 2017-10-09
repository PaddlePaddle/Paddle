# Executor Desgin Doc

## Overview

`Executor` evaluates a `ProgramDesc`. Essentially, it instantializes Variables and Operators, then run all the operators

```c++
void Executor::Run(const ProgramDesc& pdesc, Scope* scope) {
  auto& block = pdesc.blocks(0);
  auto& device = device_contexts_[0];

  // Instantiate all the vars in the global scope
  for (auto& var : block.vars()) {
    scope->NewVar(var.name());
  }

  // Decide which operator should be run
  std::vector<bool> should_run = Preprocess(pdesc);

  // Run the block
  Scope& local_scope = scope->NewScope();
  for (size_t i = 0; i < should_run.size(); ++i) {
    if (should_run[i]) {
      for (auto var : block.ops(i).outputs()) {
        for (auto argu : var.arguments()) {
          // Create variable in the local_scope
          if (local_scope.FindVar(argu) == nullptr) {
            local_scope.NewVar(argu);
          }
        }
      }
      auto op = paddle::framework::OpRegistry::CreateOp(block.ops(i));
      op->Run(local_scope, *device);
    }
  }
}
```

## Tasks

As shown above, it is not hard to simply evaluate the graph. The real problem 
is how do we actually construct the `ProgramDesc`. There are several different 
situations that we need to consider.

### 1. Init @tony @qijun

##### Problem:

Not sure which block to put init ops. Same concerns applys to `Load Model`.

##### Solution: In seperate Blocks

All `initop` and `parameter` goes to `block[0]`. Actual run starts from `block[1]`.

When user writes `a = Parameter(Variable, init)`, a init op is inserted into 
`block[0]`, and a `NOP` is inserted into  `block[1]` to substitute init op.

- Pro:
  - Init Op can be run multiple times.
  - Compatiable with current `Executor::Preprocessing`
  - Still only one `ProgramDesc`
- Con:
  - Let others know!

### 2. IO

#### 2.1 FeedOp and FetchOp

Design Doc: https://github.com/PaddlePaddle/Paddle/pull/4599

FeedOp and FetchOp in distributed environment: 
https://github.com/PaddlePaddle/Paddle/issues/4613

#### 2.2 ReaderOp and WriterOp

### 3. Backward @jiayi

Executor test case is a good place to test `backward` module, even though executor 
is not necessarily depends on `backward`. Currently exposed issue:

- Fill One: https://github.com/PaddlePaddle/Paddle/issues/4627
- Attribute map: https://github.com/PaddlePaddle/Paddle/issues/4642

### 4. Optimizer @longfei

Executor test case is a good place to test `optimizer `module, even though executor 
is not necessarily depends on `optimizer `.

### 5. RNN @chunwei

To be discussed.

- How to deal with multiple blocks
- How to deal with LoDTensor

