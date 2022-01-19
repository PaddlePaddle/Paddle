#ifndef INFRT_REWRITE
#define INFRT_REWRITE

include "paddle/infrt/dialect/infrt_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "paddle/infrt/dialect/pd_ops.td"
/*
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
def FuseMulAdd : Pat<(PD_ElementwiseAdd (PD_MatmulOp $x, $y, ConstBoolAttrFalse:$_, ConstBoolAttrFalse:$_, $alpha), $bias, $axis),
                     (PD_FusedFC $x, $y, $bias, (INFRT_createI32Attr<"1">))>;


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
                (PD_SqrtOp (PD_ElementwiseAdd $var, (PD_ConstantOp $epsilon), (INFRT_createI32Attr<"1">))),
                (INFRT_createI32Attr<"1">))),
        (PD_ElementwiseAdd
            $bias,
            (PD_MulOp 
                (PD_ElementwiseSub $bias, $mean, (INFRT_createI32Attr<"1">)),
                $coefficientW),
            (INFRT_createI32Attr<"1">)))
>;
*/
#endif // INFRT_REWRITE
