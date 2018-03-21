### Backgroud

(@kuke)

### How it works
(@kuke)

### Project structure

<p align="center">
<img src="./images/project_structure.png"/>
</p>

The project contains four important modules: 

* **fluid**: Contain wrappers for fluid related APIs. Fluid has provided some low-level APIs to parse or generate the inference model. However, directly using these low-level APIs makes the code tediously long. This module wraps low-level APIs to provide simplied interfaces.

* **onnx**: ONNX uses proto file to save computation flow and model weights. This module is responsible for parsing and generating ONNX binary model.

* **onnx_fluid**: Concepts in fluid like program, block etc. haven't direct corresponding concepts in ONNX. Even that both contains operator concept, for many operators adaption is also necessary. This module is the most important module responsible for acutal converting. Adaption for different level concepts should be provided like fluid program/block to ONNX graph, fluid operators to ONNX operators etc.

* **convert.py**: Simple top user interface. 

### Usage
The converter is very easy to use. Bi-directional conversion between fluid inference model and ONNX binary model is supported. Model validation is also provided to verify the correctness of converted model.

* fluid inference model to ONNX binary model

`python convert.py --direct fluid2ONNX --whether_do_validation True`

* ONNX binary model to fluid inference model

`python convert.py --direct ONNX2fluid --whether_do_validation True`


### Supported models
(@kuke)
