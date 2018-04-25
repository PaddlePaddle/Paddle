# Imperative Programming

This document will discuss changes needed in Paddle platform to support
imperative programming.  

Paddle provides a Python API that users typically write their training or
inference programs in.  Paddle will convert this program into a ProgramDesc, 
which is later executed in the C++ runtime.

With this flexibility, a Paddle program could contain a mix of pure Python API
and Paddle Fluid API.  To further complicate the code, the neural network logic
could be combined with the run logic (ie: Logic to run on a distributed 
system).

The mix of Python and Paddle API also makes it impossible for a transpiler to
convert the ProgramDesc to C++.

## Changes to support Imperative

Please refer to [Paddle API v4 Proposal](https://github.com/PaddlePaddle/Paddle/issues/9912)

### Startup Program

The Fluid startup programs are meant to be ran once before training starts.
Its responsible for initilization tasks, including initializing variables and
parameters.

However since startup programs are created by the users in Python, and 
execution initilized by the user in Python, this produces an issue for 
imperative.  Users will need to know when to execute the startup program,
and transpiler will have no clue how or when to execute it.

##### Proposed Solution:

With imperative programming, users may not need to directly interact with a
startup program.  To support backwards compatibility, we will still indirectly
create a startup and main program, however, before the program is executed, 
we will merge the startup program with the main program and create a new 
ProgramDesc.

There is one challenge we will need to address however:
- Since the startup and main programs are independent from each other, they
may create block id and variable names that may conflict with each other
when we merge the programs together.  We must take special care when 
re-indexing the block ids and merging variables.

### Training Loop

With imperative, the training loop will be added to the main program as an 
operator.  Since a majority of the use cases involves reading in training data
and feeding the data to the network to be trained, we chose to create an 
iterate method on reader to act as the training loop.

This iterate method will loop through the data until it has reached EOF.  
During each iteration, it will fetch training data and put them in user defined
variables.  The block within the with reader.iterate() method represents the
network to be trained.

We will need to create a `reader_loop_op`, which implement the training loop.

```Python
READER_FILE_PATH =  './data.recordio'
READER_BATCH_SIZE = 128
TRAIN_LOOP_STEPS = 100
MODEL_DIR = './model'

reader = fluid.batch_reader(file=READER_FILE_PATH, 
                            batch_size=READER_BATCH_SIZE,
                            shape=[[13], [1]], 
                            dtype=['float32', 'float32'],
                            format='recordio')

with reader.iterate() as (x, y):
  # Training Loop
  y_predict = fluid.layers.fc(input=x, size=1, act=None)

  cost = fluid.layers.square_error_cost(input=y_predict, label=y)
  avg_cost = fluid.layers.mean(cost)

  sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
  sgd_optimizer.minimize(avg_cost)

fluid.save_model(dir=MODEL_DIR)
``` 

### File Readers

We can utilize our existing recordio reader, batch reader, and shuffle reader 
ops, however we will need to make some modifications to support any additional 
methods used by our `reader_loop_op`)

### Save Model

Current a user is able to save trained parameters by calling
`fluid.io.save_persistables`.  This method will create a new Program, which
will save all the parameters to a file.  With imperative fluid, we no longer 
need to create a new Program to save variables, however we can create a simple
helper method `fluid.save_model` that will add `save`/`save_combined` operators
in the main program.

### fluid.compile

TBD

### Program.run

TBD

### Paddle Command Line

TBD
