#ifndef TRT_OPS
#define TRT_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/OpBase.td"
include "paddle/infrt/dialect/tensorrt/trt_op_base.td"

def TRT_CreateEngineOp : TRT_Op<"create_engine", [SingleBlockImplicitTerminator<"::infrt::dialect::ReturnOp">]> {
  let summary = "trt Graph Op";
  let description = [{
    Describe a tensorrt subgraph.
  }];
  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins Variadic<TRT_Tensor>:$inputs, DefaultValuedAttr<BoolAttr, "true">:$run_once);
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
