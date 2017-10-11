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
  std::vector<bool> should_run = Prune(pdesc);

  // Run the block
  Scope& local_scope = scope->NewScope();
  for (size_t i = 0; i < should_run.size(); ++i) {
    if (should_run[i]) {
      for (auto var : block.ops(i).outputs()) {
        for (auto argu : var.arguments()) {
          // Create temp variable in the local_scope
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

## Challenge

It is not hard to simply evaluate a graph. However, it is hard to determine which op should be run. Consider the following different situations.

```python
# Case 1: run foward pass.
cost_np = executor.run(target=cost)
# Case 2: run backward passing.
opts_np, _ = executor.run(target=[cost, opt])
# Case 3: run checkpointing
_ = executor.run(target=checkpoint)
```

We want to support the evaluation of both variables and operators.

## Solution

To support evaluation of operators, we add `is_target` field in the `OpDesc`.

```c++
message OpDesc {
  required string type = 3;
  repeated Var inputs = 1;
  repeated Var outputs = 2;
  repeated Attr attrs = 4;
  required bool is_target = 5 [ default = false ]; // true if the op is target
};
```

To support evaluation of variables, we add [fetch_op](https://github.com/PaddlePaddle/Paddle/pull/4599). For each variable in the `target`, we insert a `fetch_op` into the `ProgramDesc`. (Also, a user may want to overwrite a variable, so we also added [feed_op](https://github.com/PaddlePaddle/Paddle/pull/4599). )
