# RNN design
This is the design doc of the recurrent neural network operator, input instances in each mini-batch must have the same length. 

## RNN Algorithm Implementation

<p aligh="center">
<img src="./images/rnn.jpg"/>
</p>

The above diagram shows an RNN being unrolled into a full network.

There are several important concepts:

- stepnet, the network execute every time step 
- memory, a variable
- pre-memory, the value of the memory's value of the previous step time
- init_memory, the variable to help initialize memory

### step scopes
Each RNN has more than one step times, and the stepnet will be executed in every step time.
We use `Scope` to help store the contexts of all the step times:

- for each step time, create a new Scope
- create all the temporary output variables in the Scope
- execute the stepnet, and each step will have its temporary outputs

After all steps finished, RNNOp will collect the specific outputs of each step and merge them to a larger tensor.

### memory and pre-memory
a basic RNN is like:

$$
h_t = U h_{t-1} + W x_t
$$

Here, $h_t$ is time $t$'s state, $h_t$ is time $t-1$'s state, in implementation, we call the a variable that store a state memory.
In step time $t$, $h_t$ is memory, $h_{t-1}$ is pre-memory (short for previous memory).

In each step scope

- each memory variable has a corresponding pre-memory variable
- before a time step executes, copy (or make a reference) the value of previous step scope's memory to the pre-memory variable in current step scope.

### API
- void InferShape(const framework::Scope& scope) const;
  - shape check for inputs and outputs
  - infer the shapes of outputs
  
- void CreateScopes(const framework::Scope& scope) const;
  - create step scopes
  - will be called both in InferShape and Run
- void InitMemories(framework::Scope* step_scopes, bool infer_shape_mode) const;
  - make a reference to the memory in previous step scope and memory in the current one.

- void Run(const framework::Scope& scope, const platform::DeviceContext& dev_ctx) const;
  - run all the time steps.
