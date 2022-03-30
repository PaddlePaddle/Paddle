#ifndef TRT_OPS
#define TRT_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/OpBase.td"
include "paddle/infrt/dialect/tensorrt/trt_op_base.td"

include "paddle/infrt/dialect/infrt/ir/infrt_base.td"
include "paddle/infrt/dialect/phi/ir/infrt_phi_base.td"
include "paddle/infrt/dialect/pd/ir/pd_op_base.td"

def TRT_CreateEngineOp : TRT_Op<"create_engine", [SingleBlockImplicitTerminator<"::infrt::ReturnOp">]> {
  let summary = "trt CreateEngine Op";
  let description = [{
    Describe a tensorrt subgraph.
  }];
  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins Variadic<PD_Tensor>:$inputs, DefaultValuedAttr<BoolAttr, "true">:$run_once);
  let results = (outs TRT_EngineType:$engine);
}

def TRT_EngineComputeOp : TRT_Op<"compute", [NoSideEffect]> {
  let summary = "trt compute engine";
  let description = [{
    execute engine
  }];
  let arguments = (ins TRT_EngineType:$engine, Context:$context);
  let results = (outs DenseTensorList:$outputs);
}

def TRT_InspectEngineOp : TRT_Op<"inspect_engine", [NoSideEffect]> {
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

def TRT_FullyConnectedOp : TRT_Op<"FullyConnected", [NoSideEffect]> {
  let summary = "TensorRT IFullyConnectedLayer";
  let description = [{
    TensorRT IFullyConnectedLayer
  }];
  let arguments = (ins
    DenseTensor:$input_tensor,
    DenseTensor:$kernel_weights,
    DenseTensor:$bias_weights,
    SI32Attr:$out_channel_num
  );
  let results = (outs
    DenseTensor:$output_tensor
  );
}

def TRT_ConvolutionOp : TRT_Op<"Convolution", [NoSideEffect]> {
  let summary = "TensorRT IConvolutionLayer";
  let description = [{
    TensorRT IConvolutionLayer
  }];
  let arguments = (ins
    DenseTensor:$input_tensor,
    DenseTensor:$kernel_weights,
    Optional<DenseTensor>:$bias_weights,
    SI32Attr:$out_channel_num,
    I32ArrayAttr:$kernel_size,
    I32ArrayAttr:$strides,
    I32ArrayAttr:$paddings,
    StrAttr:$padding_mode,
    SI32Attr:$groups,
    I32ArrayAttr:$dilations
  );
  let results = (outs
    DenseTensor:$output_tensor
  );
}

def TRT_PoolingOp : TRT_Op<"Pooling", [NoSideEffect]> {
  let summary = "TensorRT IPoolingLayer ";
  let description = [{
    TensorRT IPoolingLayer
  }];
  let arguments = (ins
    DenseTensor:$input_tensor,
    I32Attr:$pool_type,
    I32ArrayAttr:$window_size,
    I32ArrayAttr:$strides,
    I32ArrayAttr:$paddings,
    StrAttr:$padding_mode
  );
  let results = (outs
    DenseTensor:$output_tensor
  );
}

def TRT_ElementWiseOp : TRT_Op<"ElementWise", [NoSideEffect]> {
  let summary = "TensorRT IElementWiseLayer";
  let description = [{
    
    TensorRT IElementWiseLayer.
    
  }];
  let arguments = (ins  DenseTensor:$input1, DenseTensor:$input2, SI32Attr:$elementwise_operation);

  let results = (outs DenseTensor:$output);
}

def TRT_MatrixMultiplyOp : TRT_Op<"MatrixMultiply", [NoSideEffect]> {
  let summary = "TensorRT IMatrixMultiplyLayer";
  let description = [{
    
    TensorRT IMatrixMultiplyLayer.
    
  }];
  let arguments = (ins  DenseTensor:$input1, BoolAttr:$transpose1,
                        DenseTensor:$input2, BoolAttr:$transpose2);

  let results = (outs DenseTensor:$output);
}

def TRT_ScaleOp : TRT_Op<"scale", [NoSideEffect]> {
  let summary = "TensorRT IScaleLayer";
  let description = [{

    TensorRT IScaleLayer

  }];
  let arguments = (ins  
    DenseTensor:$input_tensor,
    DefaultValuedAttr<I32Attr, "0">:$mode, 
    DenseTensor:$shift,
    DenseTensor:$scale,
    DenseTensor:$power
  );

  let results = (outs DenseTensor:$Out);
}

def TRT_MatrixMultiplOp : TRT_Op<"MatrixMultiplOp", [NoSideEffect]> {
  let summary = "TensorRT IMatrixMultiplyLayer";
  let description = [{

    TensorRT IMatrixMultiplyLayer

  }];
  let arguments = (ins  
    DenseTensor:$input1,
    DefaultValuedAttr<I32Attr, "0">:$matrix_operation1, 
    DenseTensor:$input2,
    DefaultValuedAttr<I32Attr, "0">:$matrix_operation2
  );

  let results = (outs DenseTensor:$Out);
}

def TRT_SoftMaxOp : TRT_Op<"SoftMaxOp", [NoSideEffect]> {
  let summary = "TensorRT ISoftMaxLayer";
  let description = [{

    TensorRT ISoftMaxLayer

  }];
  let arguments = (ins  
    DenseTensor:$input_tensor,
    SI32Attr:$axis
  );

  let results = (outs DenseTensor:$Out);
}

def TRT_ScaleNdOp : TRT_Op<"ScaleNd", [NoSideEffect]> {
  let summary = "TensorRT IScaleLayer";
  let description = [{

    TensorRT IScaleLayer

  }];
  let arguments = (ins  
    DenseTensor:$input_tensor,
    I32Attr:$mode,
    DenseTensor:$shift,
    DenseTensor:$scale,
    DenseTensor:$power,
    I32Attr:$axis
  );

  let results = (outs DenseTensor:$Out);
}

def TRT_ShuffleOp : TRT_Op<"Shuffle", [NoSideEffect]> {
  let summary = "TensorRT IShuffleLayer";
  let description = [{

    TensorRT IShuffleLayer

  }];
  let arguments = (ins  
    DenseTensor:$input_tensor,
    I32ArrayAttr:$reshape
  );

  let results = (outs DenseTensor:$Out);
}
#endif  // TRT_OPS
