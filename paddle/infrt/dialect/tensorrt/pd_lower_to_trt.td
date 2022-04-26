#ifndef PD_LOWER_TO_TRT
#define PD_LOWER_TO_TRT

include "mlir/Interfaces/SideEffectInterfaces.td"
include "paddle/infrt/dialect/infrt/ir/infrt_base.td"
include "paddle/infrt/dialect/pd/ir/pd_ops.td"
include "paddle/infrt/dialect/tensorrt/trt_ops.td"

class TRT_createNvinferEnumAttr<string enum_type, string enum_value> : NativeCodeCall<
    "infrt::trt::createNvinferEnumAttr<STRING_TO_ENUM_TYPE(" # enum_type # ")>($_builder, STRING_TO_ENUM_VALUE(" # enum_type # "::" # enum_value # "))">;

def PD2TRT_Matmul_Lower : Pat<
        (PD_MatmulOp $X, $Y, $transpose_X, $transpose_Y, ConstantAttr<F32Attr, "1.0">),
        (TRT_MatrixMultiplyOp $X, $transpose_X, $Y, $transpose_Y)>;

def PD2TRT_ElementwiseAdd_Lower : Pat<
        (PD_Elementwise_addOp $X, $Y, $_),
        (TRT_ElementWiseOp $X, $Y, (TRT_createNvinferEnumAttr<"nvinfer1::ElementWiseOperation", "kSUM">))>;

def PD2TRT_Relu_Lower : Pat<
        (PD_ReluOp $X),
        (TRT_ActivationOp $X, (TRT_createNvinferEnumAttr<"nvinfer1::ActivationType", "kRELU">), (INFRT_createF32Attr<"0.0">), (INFRT_createF32Attr<"0.0">))>;

def PD2TRT_Relu6_Lower : Pat<
        (PD_Relu6Op $X, $threshold),
        (TRT_ActivationOp $X, (TRT_createNvinferEnumAttr<"nvinfer1::ActivationType", "kCLIP">), (INFRT_createF32Attr<"0.0">), $threshold)>;

def createTRTConv2dOp : NativeCodeCall<"createTRTConv2dOp($_builder, $0.getDefiningOp(), $1, $2)">;

def PD2TRT_Conv2d_Lower : Pat<
        (PD_Conv2dOp:$old_value $Input, $Filter, $strides, $paddings, $padding_algorithm, $groups, $dilations, $data_format),
        (createTRTConv2dOp $old_value, $Input, $Filter)>;

def createTrtPoolingOp : NativeCodeCall<"::infrt::trt::CreatePaddleTrtPoolingOp($_builder, $0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10)">;
def PD2TRT_Pooling_Lower : Pat<
        (PD_Pool2dOp $Input, $pooling_type, $ksize, $global_pooling, $strides, $paddings, $exclusive, $adaptive, $ceil_mode, $data_format, $padding_algorithm),
        (createTrtPoolingOp $Input, $pooling_type, $ksize, $global_pooling, $strides, $paddings, $exclusive, $adaptive, $ceil_mode, $data_format, $padding_algorithm)>;

def PD2TRT_MatrixMultipl_Lower : Pat<
        (PD_MulOp $Input1, $Input2, $x_num_col_dims, $y_num_col_dims),
        (TRT_MatrixMultiplOp $Input1, (INFRT_createI32Attr<"0">)/*kNONE*/, $Input2, (INFRT_createI32Attr<"0">)/*kNONE*/)>;

def PD2TRT_SoftMax_Lower : Pat<
        (PD_SoftmaxOp $Input, $axis, $_),
        (TRT_SoftMaxOp $Input, $axis)>;

// pd.matmul_v2 + pd.elementwise_add -> trt.fc
def createTrtFcOp : NativeCodeCall<"::infrt::trt::createTrtFcOp($_builder, $0, $1, $2, $3)">;
def PD2TRT_Fc_Lower : Pat<
        (PD_Elementwise_addOp:$elt_out (PD_Matmul_v2Op $X, $Y, $trans_x, $trans_y), $elt_y, $axis),
        (createTrtFcOp $X, $Y, $elt_y, $elt_out)>;

def PD2TRT_Flatten_contiguous_range_Lower : Pat<
        (PD_Flatten_contiguous_rangeOp $input, $start_axis, $end_axis),
        (TRT_ShuffleOp $input, $start_axis, $end_axis)>;
#endif // PD_LOWER_TO_TRT
