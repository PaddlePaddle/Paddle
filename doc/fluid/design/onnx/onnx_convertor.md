### Backgroud

[ONNX (Open Neural Network Exchange)](https://github.com/onnx/onnx) bridges different deep learning frameworks by providing an open source format for models. The models trained in other frameworks can be converted into the ONNX format to execute inference by utilizing the built-in operators in ONNX. With the converse conversion, different frameworks can share any models supported by ONNX in pinciple. Now most mainstream frameworks have joined the ONNX community, e.g. Caffe2, TensorFlow, and MXNet etc. And there is a trendency that more and more vendors begin to support ONNX or even choose ONNX as the only machine learning engine in their devices. 

Therefore, it is necessary to enable the conversion between PaddlePaddle and ONNX. This design doc aims to implement the convertor, mainly for the ONNX conversion of models in Fluid and possibly including some important models in V2 format in the future. A complete convertor should be bidirectional, but  considering the importance, the conversion from Fluid to ONNX will be implemented preferentially.


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


### Supported models

Potential risks may come from the conversion of sequence-related models, including the LodTensor, ```if/else``` and ```while``` operator.
So a good choice is to focus on some important feedforward models first, then implement some simple recurrent models.
 
- Feedforward models: common models selected in PaddleBook, e.g. VGG, ResNet and some other models proposed by application teams.
- Recurrent models: language model, stacked LSTMs etc.
