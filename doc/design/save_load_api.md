# Design of Save/Load API

## Problem

We now have three save interface in [fluid](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/fluid/io.py):

- `save_vars` for saving a list of variables
- `save_params` for saving all parameters 
- `save_persistables` for saving checkpoint

And `save_vars` interface is not exposed to users directly since users can not access a variable created inside a layer. We can use `save_persistables` to do checkpoint when training. And we expect `save_params` can work in saving model for inference. 

But `save_params` and `save_persistables` are not enough to cover all cases. For example, if we want to save a model for inference network with BatchNorm layer, global mean variable is needed, whereas momentum variable is not needed. Both `save_params` and `save_persistables` are not satisfied.

We need to design a more user-friendly save/load API.

## Analysis

Let's make some analysis on the scenarios where save and load API are called.

We usually use save API in two cases:

- make checkpoint while training
- save model for inference 


And load API is a little complex, which may have three cases:

- load the checkpoint to recover training
- transfer learning. load some parameters from checkpoint and random some parameters, and do fine-tuning.
- load inference model on mobile or on-line server. 


## Solution


### New Initializer

We now have some initializers, such as UniformInitializer/NormalInitializer which will call random operator to set value for parameters in start-up program. Since we need set parameter value from local file, we should add another initializer named LocalFileInitializer, which calls load operator.

### Default Keys

We will have a default global maps to track related variables in a program. One key is "INFERENCE", and another key is "CHECKPOINT". In the construction of a program, every variable has to be add to right slot.

A save for inference example is as following: 

```
inference_vars = VarMap["INFERENCE"]
save_vars(executor, dirname, program, inference_vars)
```

And a load checkpoint example:

```
inference_vars = VarMap["CHECKPOINT"]
load_vars(...)
```

### Name Matching

Fluid now provides unique name generator to set name for each variable. And if the program is executed twice, the name of each variable will not be changed. 
However, if users have changed the topology of program, the name of variable would be changed. We have to find an appropriate to handle this. 

- TODO


### Inference Special

When loading a inference model for serving, we usually have to provide C/C++ interface. So, load API will be both implemented in Python and C++. Let's see if we can do some work to simplify this.s
