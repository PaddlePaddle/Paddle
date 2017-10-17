# Initialization of Parameters

## Motivation

PaddlePaddle needs a way to init parameters.

## Challenge

1. Init operator should be run once and only one; otherwise, every iteration would clear the parameters.
1. Init operator should not be run during training.

## Solution

### Solution 1: Two seperate `ProgramDesc`.

- Pros:
  - Simple and straight forward
- Cons:
  - Two seperate `ProgramDesc` doesn't match one-program-desc design

### Solution 2: Add run_once attribute for initialization related operators.

PR link: https://github.com/PaddlePaddle/Paddle/pull/4802/files

- Pros:
  - Simple and straight forward
- Cons:
  - Not compatible with current executor design. Because executor creates Op instances on the fly at `Executor::Run()`.

### Solution 3: Remove Init operator during Prune

Use regular expression to filter out init operators.

API at prune.cc
```c++
Prune(const ProgramDesc& input, ProgramDesc* output, const PruneInfo& prune_msg);
```

Add protobuf message
```protobuf
message PruneInfo {
  repeated string re = 3; // regular expression, e.g. "init_*".
}
```

- Pros:
  - Extendible

