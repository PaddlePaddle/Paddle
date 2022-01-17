#ifndef INFRT_TEST_OP_KERNEL_MAPPING_REWRITE
#define INFRT_TEST_OP_KERNEL_MAPPING_REWRITE

include "paddle/infrt/dialect/infrt_base.td"
include "paddle/infrt/dialect/pd_ops.td"
include "paddle/infrt/dialect/pd_test_kernels.td"

def PDKEL_Matmul_to_CPU : Pat<
          (PD_MatmulOp $x, $y, $transpose_x, $transpose_y, $alpha),
          (PDKEL_MatMulOp $x, $y, $transpose_x, $transpose_y, $alpha, (INFRT_createBoolAttr<"false">))>; // use_mkldnn = false

#endif // INFRT_TEST_OP_KERNEL_MAPPING_REWRITE
