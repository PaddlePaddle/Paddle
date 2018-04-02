### Background

[ONNX (Open Neural Network Exchange)](https://github.com/onnx/onnx) bridges different deep learning frameworks by providing an open source graph format for models. The models trained in other frameworks can be converted into the ONNX format to execute inference by utilizing the built-in operators in ONNX. With the inverse conversion, different frameworks can share any models supported by ONNX in principle. Now most mainstream frameworks have joined the ONNX community, e.g. Caffe2, TensorFlow, and MXNet etc. And there is a tendency that more and more vendors begin to support ONNX or even choose ONNX as the only machine learning engine in their devices.

Therefore, it is necessary to enable the conversion between PaddlePaddle and ONNX. This design doc aims to implement the convertor, mainly for the ONNX conversion of models in Fluid and possibly including some important models in V2 format in the future. A complete convertor should be bidirectional, but considering the importance, the conversion from Fluid to ONNX will be implemented preferentially.

One thing that makes it doable in Fluid's case is the use of a static IR - the `ProgramDesc` - as opposed to a dynamic graph, as created in the cases of frameworks like PyTorch.


### How it works

As the first step, Fluid must cover [all the listed operators](https://github.com/onnx/onnx/blob/master/docs/Operators.md) in ONNX. The complement is being carried out and only a few minor operators need to be newly added or enhanced, which would not postpone the convertor and the test of common models.

About the convertor, several things need to be considered:

- OP-level conversion
   - How to map the inputs, attributes, weights, and outputs each operator.
- Data type mapping
- Network representation adapation
   - The model in Fluid is represented by nested `Block`, how to parse and reconstruct it in ONNX graph format, and vice versa;

- Model validation
   - To assure the correctness of conversion. A simple way may be to generate some dummy data as the input and compare the inference results.
- Long term support
   - As ONNX keeps evolving, a mechanism to make sure long term support is needed.

### Project structure

<p align="center">
<img src="./images/project_structure.png"/>
</p>

The project contains four important parts:

* **fluid**: The directory that contains wrappers for fluid related APIs. Fluid has provided some low-level APIs to parse or generate the inference model. However, directly using these low-level APIs makes the code tediously long. This module wraps low-level APIs to provide simplied interfaces.

* **onnx**: ONNX uses protobuf to save computation flow and model weights. This directory consists of scripts responsible for parsing and generating an ONNX binary model.

* **onnx_fluid**: Concepts in fluid like ```program```, ```block``` etc. don't have direct corresponding concepts in ONNX. Even though both contain the operator concept, the adaption is also necessary for many operators. This directory consists of the most important modules responsible for acutal converting. Adaption for different level concepts should be provided like fluid ```program/block``` to ONNX graph, fluid operators to ONNX operators etc.

* **convert.py**: The interface exposed to users.

### Usage
The converter is designed to very easy-to-use. Bidirectional conversion between Fluid inference model and ONNX binary model is supported. Model validation is also provided to verify the correctness of converted model.

* Fluid inference model to ONNX binary model

```
python convert.py --input <fluid inference model> --output <ONNX model> --to_validate True
```

The conversion and model validation will be completed consecutively, finally output a readable model structure description. And for the converse conversion, users only need to exchange the input and output.


### Challenges and mitigation

#### Cycles

Cycles are unsupported in ONNX. In Paddle, the `while` op is the most prominent example of a cycle.

*Resolution*: We won't support models with `while`s which can't be substituted until ONNX adds support for such ops.

#### Sequences

Sequence processing operators like `sequence_expand`, `sequence_reshape`, `sequence_concat`, and `sequence_pool` are not supported by ONNX as well, because they do not support non-padded datatypes like LoDTensors.

*Resolution*: Since the runtimes using our ONNX exported graphs won't be using LoDTensors in the first place, such sequence operators should be mapped to ONNX ops that will do the necessary transposing ops with the knowledge of the padding and shape of the Tensors.

#### Ops that can't easily be mapped

There are ops that just aren't possible to map today:

**Control flow operators**

Paddle supports control flow ops like `If/Else` and `Switch` (if we ignore the CSP operations like `select` for now). ONNX has `If` support in the experimental phase.

*Resolution*: Map Paddle's `If/Else` to ONNX's `If`, but ignore other control flow operators until ONNX brings support for them.


**Non-existent in Fluid**

There are several ONNX operators that are not available in Fluid today, e.g. `InstanceNormalization`, `RandomUniform`, `Unsqueeze`, etc.

*Resolution*: For the initial phase, we can choose to not support ops that our models don't care for and are subsequently not available in Fluid. However, for ops that we think might be necessary for Fluid users also, we must implement them on our side and support the ONNX conversion to them. This list is TBD.


**Concurrency**

ONNX does not have any considerations for concurrency right now.

*Resolution*: There are two ways to approach this:

a. We choose to not support concurrent models.
b. We only support `go_op`s (basically threads) shallowly. This could mean that we enqueue `go_op` ops prior to gradient calculations OR even prior to the entire graph, and that's it - since `go_op`s do not have support for backprop anyways. One of the core target use cases of `go_op`: batch reading - can be handled through this approach.


**Overloaded in Fluid**

There are ops in ONNX whose job can't be accomplished by a single corresponding Paddle operator (e.g. ), but a collection of operators.

*Resolution*: Chain multiple Paddle operators.


#### Lack of LoDTensors

As stated above, ONNX only supports simple Tensor data.

(...)

TBD


#### Reconstruction from deprecated ONNX ops

For higher-level Fluid ops, such as a few offered by the `nn` layer that do not have direct corresponding mappings but can be converted to ONNX by chaining a series of ops without cycles, it would be useful to map them back to the higher-level Fluid ops once converted back from the deprecated ONNX graphs.

*Resolution*: Graphs that have the deprecation from Paddle -> ONNX. When converting back from ONNX, if we encounter the identical graphs by doing a forward search, we can replace the subgraphs with the matching ONNX op.


### Supported models

Potential risks may come from the conversion of sequence-related models, including the LodTensor, ```if/else``` and ```while``` operator.
So a good choice is to focus on some important feedforward models first, then implement some simple recurrent models.

- Feedforward models: common models selected in PaddleBook, e.g. VGG, ResNet and some other models proposed by application teams.
- Recurrent models: language model, stacked LSTMs etc.
