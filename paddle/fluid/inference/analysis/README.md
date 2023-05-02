# Inference Analysis

The `inference/analysis` module is used to analyze and optimize the inference program,
it references some philosophy from `LLVM/analysis`,
and make the various optimization features be pluggable and co-exist in a pipeline.

We borrowed some concepts from LLVM, such as

- [Pass](../../framework/ir/pass.h)es to implement optimization that traverse the inference program,
- [Graph](../../framework/ir/graph.h) to represent the data flow graph built from a program,
- [PassManager](./ir_pass_manager.h) to manage a sequence of `Pass`es over a graph.

There are some other basic concepts here

- [Node](../../framework/ir/node.h), the node in a `Graph`,
  - `Function`, the Operator in Fluid,
  - `Value`, the Variable in Fluid;
- [Argument](./argument.h), the argument that treat as the input and output of all `Pass`es in the pipeline,

## How it works

The `inference/analysis` module make all the passes in a pipeline, and works in such way:

1. Build a `Graph` from a Fluid inference ProgramDesc,
2. Call the middle passes one by one, the same `Graph` is passed across all the passes,
3. Transform a new ProgramDesc from the modified `Graph`.

The new optimization features can be added as an independent `Pass` and controlled by gflags,
each pass will generate unified debug information or visualization for better debugging.

## Supported Passes

### `FluidToDataFlowGraphPass`
Transform the fluid `ProgramDesc` to a `DataFlowGraph` to give an abstract representation for all the middle passes,
this should be the first pass of the pipeline.

### `DataFlowGraphToFluidPass`
Generate a final `ProgramDesc` from a data flow graph, this should be the last pass of the pipeline.

### `TensorRTSubgraphNodeMarkPass`
Mark the `Node` that are supported by TensorRT,
this pass will generate a visualization file which can be used for debugging.

### `TensorRTSubGraphPass`
Split the sub-graph that are can be accelerated by TensorRT.

### `DFG_GraphvizDrawPass`
This pass is just for debug, it will visualize the `DataFlowGraph` using the [graphviz](http://www.graphviz.org) tool.

It can be used as a helper class that draws the modified graph after each pass.

## Utilities

There is some helper legacy/function/class for analysis.

- [dot.h](./dot.h) give a easy to use interface for generating `DOT` codes,
- [graph_traits.h](../../framework/ir/graph_traits.h) contains the interfaces of the graph traversal algorithms, it uses `iterator`to make the algorithms easy to share across different passes,
there are some implementations in  [graph_helper.cc](../../framework/ir/graph_helper.cc) , such as BFS and DFS..
