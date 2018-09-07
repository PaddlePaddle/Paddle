# select_op Design

## Introduction

In golang, the [**select**](https://golang.org/ref/spec#Select_statements)
statement lets a goroutine wait on multiple communication operations at the
same time. The **select** blocks until one of its cases can run, then
executes the case.  If multiple cases are ready to run, then one case is
choosen at random to be executed.

With the introduction of CSP for Paddle, we mimic this behavior by
creating a ***select_op***.

## How to use it

The **select_op** is available as a c++ operator.  However most users
will prefer to use the much simplier Python API.

- **fluid.Select()**: Creates a select operator and adds it to the current
block within the main program.  Also creates a sub block and adds it to the
main program.  This sub block is used to hold all variables and operators
used by the case statements.

Within the select block, users can add cases by
calling **select.case** or **select.default** method.

- **fluid.Select.case(channel_action, channel, result_variable)**: Represents
a fluid channel send/recv case.  This method creates a SelectCase block
guard and adds it to the Select block.  The arguments into this method tells
the select which channel operation to listen to.

- **fluid.Select.default()**: Represents the fluid default case.  This default
case is executed if none of the channel send/recv cases are available to
execute.

**Example:**
```
ch1 = fluid.make_channel(dtype=core.VarDesc.VarType.LOD_TENSOR)
quit_ch = fluid.make_channel(dtype=core.VarDesc.VarType.LOD_TENSOR)

x = fill_constant(shape=[1], dtype=core.VarDesc.VarType.INT32, value=0)
y = fill_constant(shape=[1], dtype=core.VarDesc.VarType.INT32, value=1)

while_cond = fill_constant(shape=[1], dtype=core.VarDesc.VarType.BOOL, value=True)
while_op = While(cond=while_cond)    

with while_op.block():
    with fluid.Select() as select:
        with select.case(fluid.channel_send, channel, x):
            # Send x, then perform Fibonacci calculation on x and y
            x_tmp = fill_constant(shape=[1], dtype=core.VarDesc.VarType.INT32, value=0)
            assign(input=x, output=x_tmp)
            assign(input=y, output=x)
            assign(elementwise_add(x=x_tmp, y=y), output=y)
        with select.case(fluid.channel_recv, quit_channel, result2):
            # Exit out of While loop
            while_false = fill_constant(shape=[1], dtype=core.VarDesc.VarType.BOOL, value=False)
            helper = layer_helper.LayerHelper('assign')
            helper.append_op(
                type='assign',
                inputs={'X': [while_false]},
                outputs={'Out': [while_cond]})
```

## How it Works

### Program Description

```
blocks {
  idx: 0
  ...
  // Create "case_to_execute" variable
  ops {
    outputs {
      parameter: "Out"
      arguments: "fill_constant_110.tmp_0"
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
      f: -1.0
    }
    attrs {
      name: "shape"
      type: INTS
      ints: 1
    }
    attrs {
      name: "dtype"
      type: INT
      i: 2
    }
  }
  // Create "select" operator.
  // inputs:
  //   X: All input variables used by operators within the select block
  //   case_to_execute: Variable filled in by select_op when it determines
  //     which case to execute.
  //  
  // outputs:
  //   Out: All output variables referenced by operators within select block.
  //
  // attrs:
  //   sub_block: The block id containing the select "cases"
  //   cases:  Serialized list of all cases in the select op.
  //     Each case is serialized as: '<index>,<type>,<channel>,<value>'
  //     where type is 0 for default, 1 for send, and 2 for receive.
  //     No channel and values are needed for default cases.
  ops {
    inputs {
      parameter: "X"
      arguments: "fill_constant_103.tmp_0"
      arguments: "fill_constant_104.tmp_0"
    }
    inputs {
      parameter: "case_to_execute"
      arguments: "fill_constant_110.tmp_0"
    }
    outputs {
      parameter: "Out"
      arguments: "fill_constant_110.tmp_0"
    }    
    type: "select"
    attrs {
      name: "sub_block"
      type: BLOCK
      block_idx: 1
    }
    attrs {
      name: "cases"
      type: STRINGS
      strings: "0,1,channel_101,fill_constant_109.tmp_0"
      strings: "1,2,channel_102,fill_constant_108.tmp_0"
    }
  }
  ...
}
```

The python select API will add the **select_op** to the current block.  In addition, it will
iterate through all it's case statements and add any input variables required by case statements
into **X**.  It will also create a temp variable called **case_to_execute**.  This variable is
filled in by the select_op after it has completed processing the case statements.

If there are no available cases to execute (ie: all cases are blocked on channel operations, and
there is no default statement), then the select_op will block the current thread.  The thread will
unblock once there is a channel operation affecting one of the case statements, at which point, the
**select_op** will set the **case_to_execute** variable to the index of the case to execute.

Finally the select_op will call executor.run on the **sub_block**.

```
blocks {
  idx: 1
  parent_idx: 0
  ...
  // Fill a tensor with the case index (ie: 0,1,2,3,ect.)
  ops {
    outputs {
      parameter: "Out"
      arguments: "fill_constant_111.tmp_0"
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
      f: 0.0
    }
    attrs {
      name: "shape"
      type: INTS
      ints: 1
    }
    attrs {
      name: "dtype"
      type: INT
      i: 2
    }
  }
  // Create an "equal" operator to compare the case index with the "case_to_execute"
  // tensor (which was filled in by the select op).
  ops {
    inputs {
      parameter: "X"
      arguments: "fill_constant_111.tmp_0"  // case 0
    }
    inputs {
      parameter: "Y"
      arguments: "fill_constant_110.tmp_0"  // case_to_execute
    }
    outputs {
      parameter: "Out"
      arguments: "equal_0.tmp_0"
    }
    type: "equal"
    attrs {
      name: "axis"
      type: INT
      i: -1
    }
  }
  // Use the output of the "equal" operator as a condition for the "conditional_block".
  // If the condition evaluates to true, then execute the "sub_block" (which represents
  // the select case's body)
  ops {
    inputs {
      parameter: "Params"
    }
    inputs {
      parameter: "X"
      arguments: "equal_0.tmp_0"
    }
    outputs {
      parameter: "Out"
    }
    outputs {
      parameter: "Scope"
      arguments: "_generated_var_0"
    }
    type: "conditional_block"
    attrs {
      name: "is_scalar_condition"
      type: BOOLEAN
      b: true
    }
    attrs {
      name: "sub_block"
      type: BLOCK
      block_idx: 4
    }
  }
  ...
  // Repeat the above operators for each case statements inside the select body
}

```

Cases are represented by a **conditional_block operator**, whose's condition is set as the output of
equal(**case_to_execute**, **case_index**).  Since each case index is unique in this sub-block,
only one case will be executed.

### select_op flow

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/select_op_workflow.png"/><br/>
</p>

The select algorithm is inspired by golang's select routine.  Please refer to
http://www.tapirgames.com/blog/golang-concurrent-select-implementation for more information.

## Backward Pass

TODO
