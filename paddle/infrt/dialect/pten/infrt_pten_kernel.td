#ifndef PTEN_KERNEL
#define PTEN_KERNEL

include "paddle/infrt/dialect/phi/infrt_pten_tensor.td"

def PTEN_KernelDialect : Dialect {
  let name = "pten_kernel";

  let description = [{
    The PTEN Kernel dialect.
  }];

  let cppNamespace = "::infrt::pten";
}

// PTEN Kernel related ops.
class PDT_Kernel<string mnemonic, list<OpTrait> traits = []> : Op<PTEN_KernelDialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {
}

def FakeKernelOp : PDT_Kernel<"pten.matmul.host.fp32"> {
  let arguments = (ins CPU_Context:$dev_ctx, TensorType:$x, TensorType:$y, BoolAttr:$transpose_x, BoolAttr:$transpose_y);
  let results = (outs TensorType:$output);
}

#endif

