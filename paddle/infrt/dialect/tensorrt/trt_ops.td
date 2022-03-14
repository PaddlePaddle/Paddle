#ifndef TRT_OPS
#define TRT_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/OpBase.td"
include "paddle/infrt/dialect/tensorrt/trt_op_base.td"
include "paddle/infrt/dialect/pd_op_base.td"


def TRT_CreateEngineOp : TRT_Op<"create_engine", [SingleBlockImplicitTerminator<"::infrt::ReturnOp">]> {
  let summary = "trt CreateEngine Op";
  let description = [{
    Describe a tensorrt subgraph.
  }];
  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins Variadic<PD_Tensor>:$inputs, DefaultValuedAttr<BoolAttr, "true">:$run_once);
  let results = (outs TRT_EngineType:$output);
}

def TRT_ExecuteOp : TRT_Op<"execute", [NoSideEffect]> {
  let summary = "trt execute Op";
  let description = [{
    Describe a tensorrt runtime.
  }];
  let arguments = (ins TRT_EngineType:$engine, Variadic<PD_Tensor>:$inputs);
  let results = (outs Variadic<PD_Tensor>:$output);
}

def TRT_ActivationOp : TRT_Op<"Activation", [NoSideEffect]> {
  let summary = "TensorRT IActivationLayer";
  let description = [{
    
    TensorRT IActivationLayer.
    
  }];
  let arguments = (ins  PD_Tensor:$input, SI32Attr:$activation_type,
                        DefaultValuedAttr<F32Attr, "0.0">:$alpha,
                        DefaultValuedAttr<F32Attr, "0.0">:$beta);

  let results = (outs PD_Tensor:$output);
}

def TRT_ElementWiseOp : TRT_Op<"ElementWise", [NoSideEffect]> {
  let summary = "TensorRT IElementWiseLayer";
  let description = [{
    
    TensorRT IElementWiseLayer.
    
  }];
  let arguments = (ins  PD_Tensor:$input1, PD_Tensor:$input2, SI32Attr:$elementwise_operation);

  let results = (outs PD_Tensor:$output);
}

def TRT_MatrixMultiplyOp : TRT_Op<"MatrixMultiply", [NoSideEffect]> {
  let summary = "TensorRT IMatrixMultiplyLayer";
  let description = [{
    
    TensorRT IMatrixMultiplyLayer.
    
  }];
  let arguments = (ins  PD_Tensor:$input1, BoolAttr:$transpose1,
                        PD_Tensor:$input2, BoolAttr:$transpose2);

  let results = (outs PD_Tensor:$output);
}

def TRT_ScaleOp : TRT_Op<"Scale", [NoSideEffect]> {
  let summary = "TensorRT IScaleLayer";
  let description = [{
    
    TensorRT IScaleLayer.
    
  }];
  let arguments = (ins  PD_Tensor:$input, PD_Tensor:$scale,
                        PD_Tensor:$shift, PD_Tensor:$power,
                        DefaultValuedAttr<SI32Attr, "-1">:$channel_axis,
                        DefaultValuedAttr<SI32Attr, "-1">:$scale_mode);

  let results = (outs PD_Tensor:$output);
}

def TRT_SoftmaxOp : TRT_Op<"Softmax", [NoSideEffect]> {
  let summary = "TensorRT ISoftmaxLayer";
  let description = [{
    
    TensorRT ISoftmaxLayer.
    
  }];
  let arguments = (ins  PD_Tensor:$input,
                        DefaultValuedAttr<SI32Attr, "-1">:$axes);

  let results = (outs PD_Tensor:$output);
}

def TRT_ShuffleOp : TRT_Op<"Shuffle", [NoSideEffect]> {
  let summary = "TensorRT IShuffleLayer";
  let description = [{
    
    TensorRT IShuffleLayer.
    
  }];
  let arguments = (ins  PD_Tensor:$input,
                        DefaultValuedAttr<I32ArrayAttr, "{}">:$reshape_dimensions,
                        DefaultValuedAttr<BoolAttr, "true">:$zero_is_placeholder);

  let results = (outs PD_Tensor:$output);
}

#endif  // TRT_OPS
