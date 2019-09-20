#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging


def memory_optimize(input_program,
                    skip_opt_set=None,
                    print_log=False,
                    level=0,
                    skip_grads=True):
    """
    | Legacy memory optimization strategy, reduce total memory consumption by reuse variable memory between different operators.
    | Simple sample to explain the algorithm:
    
        ..  code-block:: python
        
            c = a + b  # assume this is the last time a is used
            d = b * c
         
    | since **a** will not be used anymore after **"c = a + b"**, and the size of **a** and **d** are the same, 
      we can use variable **a** to replace variable **d**, so actually we can optimize the above code to below:

        ..  code-block:: python
        
            c = a + b
            a = b * c 
          
    
    | Please notice that, in this legacy design, we are using variable **a** to replace **d** directly, which means 
      after you call this API, some variables may disappear, and some variables may hold unexpected values, like 
      the above case, actually **a** holds the value of **d** after execution. 
    
    | So to protect important variables from being reused/removed in the optimization, we provide skip_opt_set 
      to allow you specify a variable whitelist. 
      The variables in the skip_opt_set will not be affected by memory_optimize API.
    
    Note: 
        | **This API is deprecated, please avoid to use it in your new code.**
        | Does not support operators which will create sub-block like While, IfElse etc.
    
    Args:
        input_program(str): Input Program
        skip_opt_set(set): vars wil be skipped in memory optimze
        print_log(bool): whether to print debug log.
        level(int): 0 or 1, 0 means we replace a with b only when a.size == b.size, 1 means we can replace a with b if a.size <= b.size
    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            main_prog = fluid.Program()
            startup_prog = fluid.Program()

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            exe.run(startup_prog)
            fluid.memory_optimize(main_prog)

    """
    logging.warn(
        'Caution! paddle.fluid.memory_optimize() is deprecated '
        'and not maintained any more, since it is not stable!\n'
        'This API would not take any memory optimizations on your Program '
        'now, since we have provided default strategies for you.\n'
        'The newest and stable memory optimization strategies (they are all '
        'enabled by default) are as follows:\n'
        ' 1. Garbage collection strategy, which is enabled by exporting '
        'environment variable FLAGS_eager_delete_tensor_gb=0 (0 is the '
        'default value).\n'
        ' 2. Inplace strategy, which is enabled by setting '
        'build_strategy.enable_inplace=True (True is the default value) '
        'when using CompiledProgram or ParallelExecutor.\n')


def release_memory(input_program, skip_opt_set=None):
    """
    Modify the input program and insert :code:`delete_op` to early drop not used
    variables. The modification will be performed inplace.

    Notes: This is an experimental API and could be removed in next few
    releases. Users should not use this API.

    Args:
        input_program(Program): The program will be inserted :code:`delete_op`.
        skip_opt_set(set): vars wil be skipped in memory optimze
    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            # build network
            # ...
            
            # deprecated API
            fluid.release_memory(fluid.default_main_program())
    
    """
    logging.warn('paddle.fluid.release_memory() is deprecated, it would not'
                 ' take any memory release on your program')
