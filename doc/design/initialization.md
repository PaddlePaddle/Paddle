# Initialization of Parameters

## Motivation

PaddlePaddle needs a way to init parameters.

## Challenge

Initialization operators must run once and only one; otherwise, every iteration would clear the parameters.

## Proposals

### Proposal 1: Two seperate `ProgramDesc`.

The initialization part of the program in a `ProgramDesc` message, and the rest part in another.

- Pros:
  - Simple and straight forward.
- Cons:
  - Two seperate `ProgramDesc` doesn't match one-program-desc design.

### Proposal 2: Add `run_once` attribute for initialization related operators.

PR link: https://github.com/PaddlePaddle/Paddle/pull/4802/files

- Pros:
  - Simple and straight forward
- Cons:
  - Not compatible with current executor design. Because executor creates Op instances on the fly at `Executor::Run()`.

### Proposal 3: Remove Init operator during Prune

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

