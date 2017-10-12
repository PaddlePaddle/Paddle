# Prune

## Motivation

We want to support running inference, training and checkpointing in one `ProgramDesc`. We implement 
`void Prune(const ProgramDesc* input, ProgramDesc* output, int id)` function, which takes a `ProgramDesc`
and generate a pruned `ProgramDesc`.

## Challenge

Pruning need to support both variables and operators being evaluation targets. Consider the following
different situations.

```python
# Case 1: run foward pass.
cost_np = session.run(target=cost)
# Case 2: run backward passing.
opts_np, _ = session.run(target=[cost, opt])
# Case 3: run checkpointing
_ = session.run(target=checkpoint)
```

## Solution

To support evaluation of operators, we add `is_target` field in the `OpDesc`.

```c++
message OpDesc {
  required string type = 3;
  repeated Var inputs = 1;
  repeated Var outputs = 2;
  repeated Attr attrs = 4;
  required bool is_target = 5 [ default = false ];
};
```

To support evaluation of variables, we add [fetch_op](https://github.com/PaddlePaddle/Paddle/pull/4599).
For each variable in the `target`, we insert a `fetch_op` into the `ProgramDesc` with `variable` being
`fetch_op`'s input. Then we also set `fetch_op` is a target.

### Algorithm

If an operator needs to be run, it must fall into one of the following cases:

1. It is the target.
2. It is depended by some other ops, meaning its output is some other op's input.

The first case can be checked by `op_desc.is_traget()` . The second case can be implement as

```c++
bool HasDependentVar(const OpDesc& op_desc, const std::set<string>& dependent_vars) {
  for (auto& var : op_desc.outputs()) {
    for (auto& argu : var.arguments()) {
      if (dependent_vars.count(argu) != 0) {
        return true;
      }
    }
  }
  return false;
}
```



Then the whole algorithm can be implemented as the following

```c++
void Prune(const ProgramDesc& input, ProgramDesc* output, int id) {
  auto& block = input.blocks(id);
  auto& ops = block.ops();

  std::set<std::string> dependent_vars;
  std::vector<bool> should_run;
  
  // traverse the op list in reverse order
  for (auto op_iter = ops.rbegin(); op_iter != ops.rend(); ++op_iter) {
    auto& op_desc = *op_iter;
    if (op_desc.is_traget() || HasDependentVar(op_desc, dependent_vars)) {
      // insert its input to the dependency graph
      for (auto& var : op_desc.inputs()) {
        for (auto& argu : var.arguments()) {
          dependent_vars.insert(argu);
        }
      }

      should_run.push_back(true);
    } else {
      should_run.push_back(false);
    }
  }

  // since we are traversing the ProgramDesc in reverse order
  // we reverse the should_run vector
  std::reverse(should_run.begin(), should_run.end());
  
  output = input;
  auto* op_field = output.mutable_blocks(id)->mutable_ops();
  op_field->Clear();
  
  // add pruned ops to output
  for (size_t i = 0; i < should_run.size(); ++i) {
    if (should_run[i]) {
      *op_field->Add() = input.blocks(id).ops(i);
    }
  }
}
```
