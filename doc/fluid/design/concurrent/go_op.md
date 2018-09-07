# go_op Design

## Introduction

The **go_op** allows user's of PaddlePaddle to run program blocks on a detached
thread.  It works in conjuction with CSP operators (channel_send, 
channel_receive, channel_open, channel_close, and select) to allow users to
concurrently process data and communicate easily between different threads.

## How to use it

```
channel = fluid.make_channel(dtype=core.VarDesc.VarType.LOD_TENSOR)

with fluid.Go():
    # Send a tensor of value 99 to "channel" on a detached thread
    tensor = fill_constant(shape=[1], dtype='int', value=99)
    tensor.stop_gradient = True
    fluid.channel_send(channel, tensor)
    
# Receive sent tensor from "channel" on the main thread
result = fill_constant(shape=[1], dtype='int', value=-1)    
fluid.channel_recv(ch, result)  
```

The go operator can be accessed by using the fluid.Go() control flow.  This
will create a new sub block, where the user can add additional operators
to be ran on the thread.

**Note:** Since back propegation is currently not support in the go_op, users
should ensure that operators in the go block does not require gradient 
calculations.

## How it Works

Similar to other control blocks, go_op will create a sub block and add it
as a child to the current block.  Operators and variables defined in this
block will be added to the go sub_block.

In addition, the go operator will create a new child scope whose parent is
the global scope.  Please refer to [block captures](#block-captures) for more
information.

When Paddle executor runs go_op, go_op will take the sub_block and pass it to
the executor.run method (along with a newly created local scope) on a detached
thread.

An example of the generated program description is shown below.  Take note of
the **go_op** in particular.  It is added as an operator in the current 
block (in this example, block0).  The **go_op** contains a `sub_block`
attribute, which points to the id of the block that will be executed in a 
detached thread.

```
blocks {
  idx: 0
  parent_idx: -1
  vars {
    name: "return_value"
    type {
      type: LOD_TENSOR
      lod_tensor {
        tensor {
          data_type: INT64
        }
      }
    }
  }
  vars {
    name: "status_recv"
    type {
      type: LOD_TENSOR
      lod_tensor {
        tensor {
          data_type: BOOL
        }
      }
    }
  }
  ...
  ops {
    outputs {
      parameter: "Out"
      arguments: "channel"
    }
    type: "channel_create"
    attrs {
      name: "data_type"
      type: INT
      i: 7
    }
    attrs {
      name: "capacity"
      type: INT
      i: 0
    }
  }
  ops {
    inputs {
      parameter: "X"
      arguments: "channel"
    }
    type: "go"
    attrs {
      name: "sub_block"
      type: BLOCK
      block_idx: 1
    }
  }
  ops {
    inputs {
      parameter: "Channel"
      arguments: "channel"
    }
    outputs {
      parameter: "Out"
      arguments: "return_value"
    }
    outputs {
      parameter: "Status"
      arguments: "status_recv"
    }
    type: "channel_recv"
  }
  ...
}

blocks {
  idx: 1
  parent_idx: 0
  vars {
    name: "status"
    type {
      type: LOD_TENSOR
      lod_tensor {
        tensor {
          data_type: BOOL
        }
      }
    }
  }
  ...
  
  ops {
    outputs {
      parameter: "Out"
      arguments: "fill_constant_1.tmp_0"
    }
    type: "fill_constant"
    attrs {
      name: "force_cpu"
      type: BOOLEAN
      b: false
    }
    attrs {
      name: "value"
      type: FLOAT
      f: 99.0
    }
    attrs {
      name: "shape"
      type: INTS
      ints: 1
    }
    attrs {
      name: "dtype"
      type: INT
      i: 3
    }
  }
  ops {
    inputs {
      parameter: "Channel"
      arguments: "channel"
    }
    inputs {
      parameter: "X"
      arguments: "fill_constant_1.tmp_0"
    }
    outputs {
      parameter: "Status"
      arguments: "status"
    }
    type: "channel_send"
    attrs {
      name: "copy"
      type: BOOLEAN
      b: false
    }
  }
```

## Current Limitations

#### <a name="block-captures"></a>Scopes and block captures:

Paddle utilizes [scopes](./../concepts/scope.md) to store variables used in a
block.  When a block is executed, a new local scope is created from the parent
scope (ie: scope derived from the parent block) and associated with the new 
child block.  After the block finishes executing, then the local scope and
all associated variables in the scope is deleted.

This works well in a single threaded scenario, however with introduction of
go_op, a child block may continue to execute even after the parent block has
exited.  If the go_op tries to access variables located in the parent block's
scope, it may receive a segmentation fault because the parent scope may have
been deleted.

We need to implement block closures in order to prevent access to parent
scope variables from causing a segmentation fault.  As a temporary workaround,
please ensure that all variables accessed in the go block is not destructed
before it is being accessed.  Currently, the go_op will explicitly enforce 
this requirement and raise an exception if a variable could not be found in 
the scope.

Please refer to [Closure issue](https://github.com/PaddlePaddle/Paddle/issues/8502)
for more details.

#### Green Threads

Golang utilizes `green threads`, which is a mechnism for the runtime library to 
manage multiple threads (instead of natively by the OS).  Green threads usually
allows for faster thread creation and switching, as there is less overhead
when spawning these threads.  For the first version of CSP, we only support
OS threads.


#### Backward Propegation:

go_op currently does not support backwards propagation.  Please use go_op with
non training operators.
