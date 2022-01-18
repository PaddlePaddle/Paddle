#ifndef PD_TEST_KERNELS
#define PD_TEST_KERNELS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/OpBase.td"
include "paddle/infrt/dialect/pd_op_base.td"


// This is just for test case.
def PD_KELMatmulOp : PD_Op<"matmul_kernel"> {
  let summary = "matmul kernel op";
  let description = [{}];

  let arguments = (ins PD_Tensor:$X, PD_Tensor:$Y,
                   DefaultValuedAttr<BoolAttr, "false">:$transpose_X,
                   DefaultValuedAttr<BoolAttr, "false">:$transpose_Y,
                   DefaultValuedAttr<F32Attr, "1.0">:$alpha,
                   DefaultValuedAttr<BoolAttr, "false">:$use_mkldnn
                   );

  let results = (outs PD_Tensor:$Out);
}

#endif
