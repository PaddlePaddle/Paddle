#ifndef PTEN_KERNEL
#define PTEN_KERNEL

include "paddle/infrt/dialect/pten/infrt_pten_tensor.td"

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

#endif

