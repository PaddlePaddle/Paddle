#ifndef INFRT_REWRITE
#define INFRT_REWRITE

include "paddle/infrt/dialect/pd/ir/pd_ops.td"
include "paddle/infrt/dialect/phi/ir/infrt_phi_tensor.td"

def Conv2dBnFuse_CreateFilter : NativeCodeCall<
  "::infrt::dialect::Conv2dBnFuse_CreateFilter($_builder, $_loc, $0, $1, $2, $3)">;

def Conv2dBnFuse_CreateBias : NativeCodeCall<
  "::infrt::dialect::Conv2dBnFuse_CreateBias($_builder, $_loc, $0, $1, $2, $3, $4)">;

def Conv2dBnFuse: Pat<(PD_Batch_normOp 
                        (PD_Conv2dOp $Input, $Filter, $strides, $paddings, $padding_algorithm,
                        $groups, $dilations, $data_format),
                        $Scale, $Bias, $Mean, $Variance, $MomentumTensor, $momentum, $epsilon,
                        $data_layout),
                      (PD_Elementwise_addOp 
                        (PD_Conv2dOp $Input, 
                        (Conv2dBnFuse_CreateFilter $Filter, $Variance, $Scale, $epsilon),
                        $strides, $paddings, $padding_algorithm, $groups, $dilations, $data_format),
                        (Conv2dBnFuse_CreateBias $Mean, $Variance, $Scale, $Bias, $epsilon),
                        (INFRT_createSI32Attr<"1">))>;

#endif // INFRT_REWRITE
