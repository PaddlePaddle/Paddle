#ifndef PD_LOWER_TO_TRT
#define PD_LOWER_TO_TRT

include "mlir/Interfaces/SideEffectInterfaces.td"
include "paddle/infrt/dialect/infrt/ir/infrt_base.td"
include "paddle/infrt/dialect/pd/ir/pd_ops.td"
include "paddle/infrt/dialect/tensorrt/trt_ops.td"

class TRT_createNvinferEnumAttr<string value> : NativeCodeCall<
    "infrt::trt::createNvinferEnumAttr($_builder, STRING_TO_ENUM(" # value # "))">;

def PD2TRT_Matmul_Lower : Pat<
        (PD_MatmulOp $X, $Y, $transpose_X, $transpose_Y, ConstantAttr<F32Attr, "1.0">),
        (TRT_MatrixMultiplyOp $X, $transpose_X, $Y, $transpose_Y)>;

def PD2TRT_ElementwiseAdd_Lower : Pat<
        (PD_Elementwise_addOp $X, $Y, ConstantAttr<SI32Attr, "-1">),
        (TRT_ElementWiseOp $X, $Y, (TRT_createNvinferEnumAttr<"nvinfer1::ElementWiseOperation::kSUM">))>;

def PD2TRT_Relu_Lower : Pat<
        (PD_ReluOp $X),
        (TRT_ActivationOp $X, (TRT_createNvinferEnumAttr<"nvinfer1::ActivationType::kRELU">), (INFRT_createF32Attr<"0.0">), (INFRT_createF32Attr<"0.0">))>;

def PD2TRT_Relu6_Lower : Pat<
        (PD_Relu6Op $X, $threshold),
        (TRT_ActivationOp $X, (TRT_createNvinferEnumAttr<"nvinfer1::ActivationType::kCLIP">), (INFRT_createF32Attr<"0.0">), $threshold)>;

#endif // PD_LOWER_TO_TRT
