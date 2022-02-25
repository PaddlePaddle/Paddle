#ifndef PHI_KERNEL
#define PHI_KERNEL

include "paddle/infrt/dialect/phi/infrt_phi_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "paddle/infrt/dialect/infrt_base.td"

class DenseTensorType<string target, string layout, string precision> :
    Type<CPred<"$_self.isa<::infrt::DenseTensorType>()">, "!infrt.densetensor type", "::infrt::DenseTensorType">,
    BuildableType<"$_builder.getType<::infrt::DenseTensorType,::infrt::TargetType,::infrt::LayoutType, ::infrt::PrecisionType >(" #"::infrt::TargetType::"#target#","#"::infrt::LayoutType::"#layout#","#"::infrt::PrecisionType::"#precision#")" >;

def PHI_CPUKernelDialect : Dialect {
  let name = "phi_cpu_kernel";

  let description = [{
    The PHI CPU Kernel dialect.
  }];

  let cppNamespace = "::infrt::phi";
}

def PHI_GPUKernelDialect : Dialect {
  let name = "phi_gpu_kernel";

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
