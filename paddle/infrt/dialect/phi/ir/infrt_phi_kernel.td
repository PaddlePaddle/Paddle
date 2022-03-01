#ifndef PHI_KERNEL
#define PHI_KERNEL

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "paddle/infrt/dialect/infrt_base.td"
include "paddle/infrt/dialect/phi/ir/infrt_phi_base.td"

class DenseTensorType<string target, string layout, string precision> :
    Type<CPred<"$_self == ::infrt::DenseTensorType::get($_self.getContext(),::infrt::TargetType::"#target#", ::infrt::LayoutType::"#layout#", ::infrt::PrecisionType::"#precision#")">, "!infrt.densetensor type", "::infrt::DenseTensorType">;

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
