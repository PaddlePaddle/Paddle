#ifndef PHI_KERNEL
#define PHI_KERNEL

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "paddle/infrt/dialect/infrt/ir/infrt_base.td"
include "paddle/infrt/dialect/phi/ir/infrt_phi_base.td"

def PHI_CPUKernelDialect : Dialect {
  let name = "phi_cpu";

  let description = [{
    The PHI CPU Kernel dialect.
  }];

  let cppNamespace = "::infrt::phi";
}

def PHI_GPUKernelDialect : Dialect {
  let name = "phi_gpu";

  let description = [{
    The PHI GPU Kernel dialect.
  }];

  let cppNamespace = "::infrt::phi";
}

// PHI Kernel related ops.
class PDTCPU_Kernel<string mnemonic, list<OpTrait> traits = []> : Op<PHI_CPUKernelDialect, mnemonic, !listconcat(traits, [PhiOpTrait, IsolatedFromAbove])> {
}

// PHI Kernel related ops.
class PDTGPU_Kernel<string mnemonic, list<OpTrait> traits = []> : Op<PHI_GPUKernelDialect, mnemonic, !listconcat(traits, [PhiOpTrait, IsolatedFromAbove])> {
}

#endif
