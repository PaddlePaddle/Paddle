# Utilize Engines to Accelerate Inference

The inference phase need to support some special hardware for acceleration, 
such as GPU, FPGA, and ARM.
Special softwares power some of these hardwares and the inner states are hidden, for example, the TensorRT is released by NVidia to improve the inference performance on GPUs, it takes a computation graph as input, 
optimize and execute it, but the users can't directly modify its internal logics. 

In other words, these software acts like a black box, the external logics prepare its inputs, execute it and process its output. 
In the Paddle inference module, we call such software an engine, and the inference phase will partition sub-blocks(sub graph) and execute them on the engines to improve performance.

## Use Engines to Execute Sub-blocks

Compared to Paddle Fluid, the engines covers limited number of operators and can only power several kinds of models. In other words, the engines can only support a part of Fluid.

The Block in Fluid acts like a computation graph and it is natural to partition the Block into several sub-blocks which are powered by several different engines.

<p align="center">

<img src="./images/inference engine.jpg"/>

</p>

It is easy to parallelize the computation by scheduling several engines on different devices, for example, the CPU and GPU engines can be dispatched in the meantime

<p align="center">

<img src="./images/parallel engine.png"/>

</p>



## Partition the Sub-blocks supported by a Specific Engine

As mentioned above, one engine can support a partition of Fluid operators, the sub-block dispatched should be composed by the operators this engine fully supports.

The Inference framework needs a mechanism to mark the sub-block and deliver it to an engine.

We use a `with-statement` to mark the sub-block as follows.

```python
with infer.power_by_engine('tensorrt'):
    o = some_op()
    o = some_op()
    ...
```

The operators inside the `infer.power_by_tensorrt` code block will combine into a sub-block and transfers to a TensorRT engine. We call this API-powered sub-block marking the way the **manul sub-block marking**, which means the users directly decide the operators runs on some engine.

For large models, it is trivial to mark all the sub-blocks, so an elicitation method is raised to make it automatic, we call this way the **sub-block automatic detection mode**.

```python
# if min_ops is set, turn on sub-block automatic detection mode
# if the more than two adjacent operators are supported by some engine, combine them to 
# a sub-block and transmit to some engine.
infer.init_subblock_optimizer(min_ops=2)

o = some_op()
o = some_op()

# one can still set one or a code block to use some specific engine
with infer.power_by_engine('X'):
    o = op1()

o = some_op()

# several different engines can be utilized in one model, the elicitation method 
# will greedily detect more adjacent operators that powered by the specified engine,
#, and partition to a larger sub-block.
with infer.power_by_engine('Y'):
    o = op2()
  
o = some_op()
```

## Transmit the sub-blocks to an Engine

The marked code blocks will be written into a `BlockDesc`, to make the inference phase more clear to support the engine execution, we break up the whole architecture into three layers:

- Frontend, the python syntax, generate the basic fluid model description with some inference customized configurations.
- Optimizer, rewrite the fluid description, such as pruning the unused operators, reuse some variable memory.
- Backend, simply execute the fluid description.

<p align="center">

<img src="images/inference architecture.png"/>

</p>

To support engines, there are following phases.

1. the user uses the APIs supplied by **Frontend**, generate a Fluid model description with some adjacent operators marked with some engine label
2. the optimizer found out the operators marked with the engine labels
   1. Extract these operators and combine them to sub-blocks
   2. Delete them from the original Fluid model description
   3. Insert a new `EngineOp` into the original Fluid model description with the sub-block set as an attribute
3. the **Backend** get the optimized Fluid description
   - the `EngineOp` is treated as normal operators
   - the Backend execute each operator one by one if in sync mode
   - the operators(especially different `EngineOp` on different devices) can execute parallelly in async mode

## How Engine works with the Fluid framework

The `EngineOp` described above is the key, it acts like normal Fluid operators, but embed with an engine. When an `EngineOp` is created, the engine inside will build a network which acts equivalent functions with the Fluid sub-block describes.

When the whole Fluid description is executed by Backend, the engine inside will run its runtime engine.

There is a tradeoff between sub-block size and the number of `EngineOp,` each `EngineOp` need a pair of input and output data format converters, those results in additional latency. 

So bigger sub-block with less `EngineOp` is better, but some Fluid operators without alternative ones in the engine will break up the big block into small subblocks, whether to execute these sub-blocks on engines or just on Fluid, that needs more consideration.

To help convert input/output data format between any Fluid operators and some `EngineOp`, a pair of `EngineInputConvertOp` and `EngineOutputConvertOp` needs to insert into the Fluid description. The reason for these converters is an operator, not a method are as follows

- the converter works between the Fluid operators and `EngineOp`s, both of their data formats might be different and need to specify, for example
  - `RNNOp -> xEngineOp`, the input is from an `LoDTensor` to `xTensor`
  - `MulOp -> xEngineOp`, the input is from an `Tensor` to `xTensor`
  - but `RNNOp -> MulOp -> xEngineOp`, the input is from an `LoDTensor` to `xTensor`
  - the `EngineOp` can not get the external operators those link to it (the `RNNOp` and `MulOp` above), so it is impossible to deduce the data's format interact with the external Fluid framework.
- the converter will result in additional overhead, to make it an operator is more clear and more flexible for further optimization.

## Engine-related Design

###  EngineOp                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

`EngineOp` is just a normal Fluid operator, which has an attribute called `subblock` to get the Fluid description about a sub-block.

### Data format convert operators

Both the `EngineInputConvertOp` and `EngineOutputConvertOp` has similar interfaces

The operator has following attributes.

1. `input_op_type` 
2. `output_op_type`

For a convert op that input from `RNNOp` and outputs to `xEngineOp`, the values of these two attributes are *RNNOp* and *xEngineOp*.

to make the implementation of the input and output combination more extensible, the functor and register should be included:

```c++
struct EngineInputConveterBase {
      // the `out` is a cuda memory that has been allocated.
      virtual void operator()(LoDTensor& in, void* out, size_t max_size) = 0;
      static void Execute(const std::string& in_op_type, const std::string& out_op_type,
                      const LoDTensor& in, void* out, size_t max_size) {
    conveters[in_op_type + "_to_" + out_op_type](in, out, max_size);
  }
  
      static std::map<std::string, EngineInputConveterBase> conveters;
};

// some specific implementations
struct RNN2xEngineConveter : public EngineInputConveterBase {
    void operator()(const LoDTensor& in, void* out, size_t max_size) override;
};

#define REGISTER_INPUT_CONVERTER(in_op_type__, out_op_type__, conveter__) \
    EngineInputConveterBase::conveters[#in_op_type__ "_to_" #out_op_type__] = conveter__();

REGISTER_INPUT_CONVERTER(RNNOp, xEngineOp, RNN2xEngineConveter);
```

The `EngineOutputConvertOp` is similar.

### Optimizer for sub-block

```c++
// The InferenceOptimizers input a program desc and output a block desc.
// Different implementations will rewrite the original program desc by different logics.
// There might be many different optimizers, such as
// - CleanUselessOptimizer
// - PruneOpOptimizer
// - SubblockToEngineOptimizer
struct InferenceOptimizer {
   virtual operator()(const ProgramDesc& desc, ProgramDesc* out) = 0;
    
    void RunALL() {
        for (auto& o : optimizers) { o(); }
    }
    
    template<typename T>
    static void Register(const T& t) { optimizers.append(t); }
    static std::vector<InfernceOptimizer> optimizers;
};

// Extract the subblock from the Program Desc, insert a xEngineOp and set its attribute
// to a sub-block description.
struct SubblockToEngineOptimizer : public InferenceOptimizer {
    virtual operator() (const ProgramDesc& desc, ProgramDesc* out) override;
};

REGISTER_INFERENCE_OPTIMIZER(SubblockToEngineOptimizer);

#define REGISTER_INFERENCE_OPTIMIZER(Optimizer__) \
   InferenceOptimizer::Register(Optimizer__());
```


