#ifdef PTEN_KERNEL
#else
#define PTEN_KERNEL

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "paddle/infrt/dialect/infrt_base.td"
include "paddle/infrt/dialect/pten/infrt_pten_base.td"

def PTEN_KernelDialect : Dialect {
  let name = "pten_kernel";

  let description = [{
    The PTEN Kernel dialect.
  }];

  let cppNamespace = "::infrt::pten";
}

// PTEN Kernel related ops.
class PDT_Kernel<string mnemonic, list<OpTrait> traits = []> : Op<PTEN_KernelDialect, mnemonic, !listconcat(traits, [PtenOpTrait, IsolatedFromAbove])> {
}

#endif
