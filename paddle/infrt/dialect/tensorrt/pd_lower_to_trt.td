#ifndef PD_LOWER_TO_TRT
#define PD_LOWER_TO_TRT

include "mlir/Interfaces/SideEffectInterfaces.td"
include "paddle/infrt/dialect/infrt/ir/infrt_base.td"
include "paddle/infrt/dialect/pd/ir/pd_ops.td"
include "paddle/infrt/dialect/tensorrt/trt_ops.td"

def PD2TRT_Matmul_Lower : Pat<
        (PD_MatmulOp $X, $Y, $transpose_X, $transpose_Y, ConstantAttr<F32Attr, "1.0">),
        (TRT_MatrixMultiplyOp $X, $transpose_X, $Y, $transpose_Y)>;

//TO DO(shangzhizhou):replace '"INFRT_createI32Attr<"0">' to enum nvinfer1::ElementWiseOperation::kSUM
def PD2TRT_ElementwiseAdd_Lower : Pat<
        (PD_Elementwise_addOp $X, $Y, ConstantAttr<SI32Attr, "-1">),
        (TRT_ElementWiseOp $X, $Y, (INFRT_createSI32Attr<"0">)/*kSUM*/)>;

//TO DO(shangzhizhou):replace '"INFRT_createI32Attr<"0">' to enum  nvinfer1::ActivationType::kRELU
def PD2TRT_Relu_Lower : Pat<
        (PD_ReluOp $X),
        (TRT_ActivationOp $X, (INFRT_createSI32Attr<"0">)/*kRELU*/, (INFRT_createF32Attr<"0.0">), (INFRT_createF32Attr<"0.0">))>;

//TO DO(shangzhizhou):replace '"INFRT_createI32Attr<"0">' to enum  nvinfer1::ActivationType::kCLIP
def PD2TRT_Relu6_Lower : Pat<
        (PD_Relu6Op $X, $threshold),
        (TRT_ActivationOp $X, (INFRT_createSI32Attr<"8">)/*kCLIP*/, (INFRT_createF32Attr<"0.0">), $threshold)>;

#endif // PD_LOWER_TO_TRT
