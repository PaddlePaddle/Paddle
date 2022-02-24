#ifndef PHI_KERNEL
#define PHI_KERNEL

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "paddle/infrt/dialect/infrt_base.td"
include "paddle/infrt/dialect/phi/infrt_phi_base.td"

def PHI_KernelDialect : Dialect {
  let name = "phi_kernel";

  let description = [{
    The PHI Kernel dialect.
  }];

  let cppNamespace = "::infrt::phi";
}

// PHI Kernel related ops.
class PDT_Kernel<string mnemonic, list<OpTrait> traits = []> : Op<PHI_KernelDialect, mnemonic, !listconcat(traits, [PhiOpTrait, IsolatedFromAbove])> {
}

def PDCK_AbsOp : PDT_Kernel<"phi.abs.host.fp32"> {
  let arguments = (ins CPU_Context:$dev_ctx, DenseTensor:$x);
  let results = (outs DenseTensor:$output);
}

#endif

