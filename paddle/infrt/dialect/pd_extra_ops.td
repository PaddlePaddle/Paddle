#ifndef PD_EXTRA_OPS
#define PD_EXTRA_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/OpBase.td"
include "paddle/infrt/dialect/pd_op_base.td"

def PD_FusedFC : PD_Op<"FC", [NoSideEffect]> {
    let summary = "Computes the Fully Connected result of two tensors";
    let description = [{
    }];

    let arguments = (ins PD_Tensor:$input, PD_Tensor:$w, PD_Tensor:$bias, DefaultValuedAttr<I32Attr, "1">:$in_num_col_dims);
    let results = (outs PD_Tensor:$out);
}

#endif
