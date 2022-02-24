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
  let arguments = (ins Variadic<TRT_Tensor>:$inputs);
  let results = (outs Variadic<TRT_Tensor>:$outputs);

}

def TRT_ActivationOp : TRT_Op<"Activation", [NoSideEffect]> {
  let summary = "TensorRT IActivationLayer";
  let description = [{
    
    TensorRT IActivationLayer.
    
  }];
  let arguments = (ins  TRT_Tensor:$input, SI32Attr:$activation_type,
                        DefaultValuedAttr<F32Attr, "0.0">:$alpha,
                        DefaultValuedAttr<F32Attr, "0.0">:$beta);

  let results = (outs TRT_Tensor:$output);
}

def TRT_ElementWiseOp : TRT_Op<"ElementWise", [NoSideEffect]> {
  let summary = "TensorRT IElementWiseLayer";
  let description = [{
    
    TensorRT IElementWiseLayer.
    
  }];
  let arguments = (ins  TRT_Tensor:$input1, TRT_Tensor:$input2, SI32Attr:$elementwise_operation);

  let results = (outs TRT_Tensor:$output);
}

def TRT_MatrixMultiplyOp : TRT_Op<"MatrixMultiply", [NoSideEffect]> {
  let summary = "TensorRT IMatrixMultiplyLayer";
  let description = [{
    
    TensorRT IMatrixMultiplyLayer.
    
  }];
  let arguments = (ins  TRT_Tensor:$input1, BoolAttr:$transpose1,
                        TRT_Tensor:$input2, BoolAttr:$transpose2);

  let results = (outs TRT_Tensor:$output);
}

#endif  // TRT_OPS
