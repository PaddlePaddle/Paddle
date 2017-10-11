# Executor Desgin Doc

## Overview

`Executor` evaluates a `ProgramDesc`. Essentially, it instantializes Variables and Operators, then run all the operators

```c++
void Executor::Run(const ProgramDesc& pdesc, Scope* scope, int block_id) {
  auto& block = pdesc.blocks(block_id);
  auto& device = device_contexts_[0];

  // Instantiate all the vars in the global scope
  for (auto& var : block.vars()) {
    scope->NewVar(var.name());
  }

  // Run the block
  Scope& local_scope = scope->NewScope();
  for (size_t i = 0; i < should_run.size(); ++i) {
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
```
