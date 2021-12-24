#ifndef TRT_OPS
#define TRT_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/OpBase.td"
include "paddle/infrt/dialect/tensorrt/trt_op_base.td"

def TRT_FetchOp : TRT_Op<"fetch", [Terminator]> {
  let summary = "TensorRT engine return operation";
  let description = [{
    The `trt.fetch` operation terminates and returns values for the
    `trt.graph` operation.
    }];

  let arguments = (ins Variadic<TRT_Tensor>:$inputs);
}

def TRT_GraphOp : TRT_Op<"graph", [SingleBlockImplicitTerminator<"FetchOp">]> {
  let summary = "trt Graph Op";
  let description = [{
    Describe a tensorrt subgraph.
  }];
  let regions = (region SizedRegion<1>:$body);
  
  let results = (outs Variadic<TRT_Tensor>:$outputs);

}
#endif  // TRT_OPS
