#ifndef PD_OP_FUSE
#define PD_OP_FUSE

include "mlir/Interfaces/SideEffectInterfaces.td" 
include "paddle/infrt/dialect/core/ir/core_base.td"
include "paddle/infrt/dialect/pd/ir/pd_ops.td"
include "paddle/infrt/dialect/pd/ir/pd_extra_ops.td"

//===----------------------------------------------------------------------===//
// This is to fuse the composition: 'Matmul o ElementwiseAdd' into 'PD_FusedFC'.
//
// We have:
//   (Matmul)      z = x * y
//   (Add)         out = z + bias 
//
// which corresponds to the following computation:
//   (FusedFC)  out = x * y + bias
// 
// while meeting the following attribute constrait:
// Matmul: transpose_x: false
//         transpose_y: false
//
// Todo:
//  1. Make the constrait more completely.
//  2. Consider the case of : out = bias + z
//===----------------------------------------------------------------------===//
def FuseMulAdd : Pat<(PD_Elementwise_addOp (PD_Matmul_v2Op $x, $y, ConstBoolAttrFalse:$_, ConstBoolAttrFalse:$_), $bias, $axis),
                     (PD_FusedFC $x, $y, $bias, (CORE_createI32Attr<"1">))>;


/*
//===----------------------------------------------------------------------===//
// This is to fuse the composition: 'FusedFC o Relu' into 'FusedRepeatedFCRelu'.
//
// We have:
//   (FusedFC)      z = fc(x, y, bias)
//   (Relu)         out = relu(z)
//
// which corresponds to the following computation:
//   (FusedRepeatedFCRelu)  out = RepeatedFCRelu(x, [y], [bias])
// 
//===----------------------------------------------------------------------===//
def FuseFCRelu : Pat<(PD_ReluOp (PD_FusedFC $x, $y, $bias, $_)),
                     (PD_FusedRepeatedFCRelu $x, (INFRT_cvtValueToValueRange $y), (INFRT_cvtValueToValueRange $bias))>;

//===----------------------------------------------------------------------===//
// This is to fold 'FusedRepeatedFCRelu' op.
//
// We have:
//   (FusedRepeatedFCRelu)      z = RepeatedFCRelu(x, [y, ...], [bias, ...])
//   (FusedRepeatedFCRelu)      out = RepeatedFCRelu(z, [y1, ...], [bias1, ...])
//
// which corresponds to the following computation:
//   (FusedRepeatedFCRelu)  out = RepeatedFCRelu(x, [y, ..., y1, ...], [bias, ..., bias1, ....])
// 
//===----------------------------------------------------------------------===//
def FuseRepeatedFCRelu2 : Pat<(PD_FusedRepeatedFCRelu (PD_FusedRepeatedFCRelu $x, $y, $bias), $y_2, $bias_2),
                     (PD_FusedRepeatedFCRelu $x, (INFRT_concatTwoValueRange $y, $y_2), (INFRT_concatTwoValueRange $bias, $bias_2))>;


//===----------------------------------------------------------------------===//
// This is to fuse the composition: 'BatchNorm o Conv' into 'Conv'
// by deriving new 'w' and 'b' for 'Conv':
//
// We have:
//   (Conv)      z = w * x + b 
//   (BatchNorm) y = scale * (z - mean) / sqrt(var + eps) + bias
//
// which corresponds to the following computation:
//   y = w_ * x + b_
// where
//   w_ = scale * w / sqrt(var + eps)
//   b_ = B + scale * (b - mean) / sqrt(var + eps)
//
//===----------------------------------------------------------------------===//
def FuseBatchNormWithConvPattern: Pat<
    (PD_BatchNormOp
        (PD_Conv2dOp $input, $filter, $bias),
        $scale, $bias_2, $mean, $var, $epsilon),
    (PD_Conv2dOp
        $input,
        (PD_MulOp $filter,
            (PD_ElementwiseDiv:$coefficientW
                $scale,
                (PD_SqrtOp (PD_ElementwiseAdd $var, (PD_ConstantOp $epsilon), (CORE_createI32Attr<"1">))),
                (CORE_createI32Attr<"1">))),
        (PD_ElementwiseAdd
            $bias,
            (PD_MulOp 
                (PD_ElementwiseSub $bias, $mean, (CORE_createI32Attr<"1">)),
                $coefficientW),
            (CORE_createI32Attr<"1">)))
>;
*/
#endif // PD_OP_FUSE
