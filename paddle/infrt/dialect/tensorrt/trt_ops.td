#ifndef TRT_OPS
#define TRT_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/OpBase.td"
include "paddle/infrt/dialect/tensorrt/trt_op_base.td"

include "paddle/infrt/dialect/infrt/infrt_ops_base.td"
include "paddle/infrt/dialect/phi/ir/infrt_phi_base.td"

def TRT_CreateEngineOp : TRT_Op<"create_engine", [SingleBlockImplicitTerminator<"::infrt::ReturnOp">]> {
  let summary = "trt CreateEngine Op";
  let description = [{
    Describe a tensorrt subgraph.
  }];
  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins Variadic<DenseTensor>:$inputs, DefaultValuedAttr<BoolAttr, "true">:$run_once);
  let results = (outs TRT_EngineType:$engine);
}

def TRT_ExecuteOp : TRT_Op<"execute", [NoSideEffect]> {
  let summary = "trt execute Op";
  let description = [{
    Describe a tensorrt runtime.
  }];
  let arguments = (ins TRT_EngineType:$engine, Variadic<TRT_Tensor>:$inputs);
  let results = (outs Variadic<TRT_Tensor>:$output);
}

def TRT_EngineCompute : TRT_Op<"compute", [NoSideEffect]> {
  let summary = "trt compute engine";
  let description = [{
    execute engine
  }];
  let arguments = (ins TRT_EngineType:$engine, Context:$context);
  let results = (outs TensorListType:$outputs);
}

def TRT_InspectEngine : TRT_Op<"inspect", [NoSideEffect]> {
  let summary = "trt inspect engine";
  let description = [{
    Show engine
  }];
  let arguments = (ins TRT_EngineType:$engine);
}

def TRT_ActivationOp : TRT_Op<"Activation", [NoSideEffect]> {
  let summary = "TensorRT IActivationLayer";
  let description = [{
    
    TensorRT IActivationLayer.
    
  }];
  let arguments = (ins  DenseTensor:$input, SI32Attr:$activation_type,
                        DefaultValuedAttr<F32Attr, "0.0">:$alpha,
                        DefaultValuedAttr<F32Attr, "0.0">:$beta);

  let results = (outs DenseTensor:$output);
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
