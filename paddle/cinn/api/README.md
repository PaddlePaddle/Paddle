The classes in this directory are the interface of group fusion pass, you can use these apis to build the stragey for group fusion.


The Class and APIs are following:

`OpGroup` : A set of op nodes, which will pass to cinn backend for generating kernel code. Two groups can fuse togather according to the rule of merging written in the passes.

`OpNode` : Map the op in the program.

`TensorNode` : Map the tensor in the program.

`Shape` : The shape infomation of tensor

`FusePassCtx` : The context is the parameter for the pass, it hold the data all you need in the pass.

`FuseHelper` : We provide some util methods such as `DetectCycleIfFuse` in fuse_helper to simplify development of pass.

| Class      | method | description |
| :--:       | :--: | :--: |
| OpGroup  | kind()| Get the Kind of group |
|            | producers()| Get producer groups of current group |
|            | consumers() | Get consumer groups of current group |
|            | WalkOpNodes(const std::function<void(const OpNode&)>& VisitOpNode) | Visit the op_nodes in the group and execute the VisitOpNode function for each OpNode |
|  |  |  |
| OpNode   | kind() | Get the Kind of op_node |
|            | inputs() | Get input tensors of op_node |
|            | outputs() | Get output tensors of op_node |
|            | GetAttr(const std::string& attr_name) | Get attribute of op_node by attr name |
|  |  |  |
| TensorNode | shape() | Get shape of tensor |
|            | producer() | Get the producer op_node of tensor |
|            | consumers() | Get the consumer op_nodes of tensor |
|  |  |  |
| Shape    | numel() | Get total number of elements in the shape |
|            | other methods are same with std::vector<int64_t> | |
|  |  |  |
| LightwareFusePassCtx | PickOpGroup() | Get the current group in the pass context |
|                      | void EnableFuse(const OpGroup& first, const OpGroup& second) | Mark the two groups which can fuse togather |
|  | fuse_helper()     | Get the fuse_helper provided by pass context  |
|  |  |  |
| InputFusePassCtx   | PickConsumersWithSameInputs() | Get all consumer groups for input tensors of graph |
|                      | void EnableFuse(const OpGroup& first, const OpGroup& second) | Mark the two groups which can fuse togather |
|  | fuse_helper()     | Get the fuse_helper provided by pass context  |
|  |  |  |
| FuseHelper | DetectCycleIfFuse(const OpGroup& first, const OpGroup& second) | Whether there is cycle in graph after fusing two groups |
