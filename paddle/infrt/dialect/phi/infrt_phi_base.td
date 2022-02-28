#ifndef PHI_BASE
#define PHI_BASE

include "mlir/IR/OpBase.td"

def PHI_Dialect : Dialect {
  let name = "phi";

  let description = [{
    The PHI host dialect.
  }];

  let cppNamespace = "::infrt::phi";
}

class AllocatorTypeOf<string place, list<Trait> traits=[]>:
    TypeDef<PHI_Dialect, place # "Allocator", traits> {
    let summary = !strconcat("!phi.allocator_", place, " type");
}

class ContextTypeOf<string place, list<Trait> traits=[]>:
    TypeDef<PHI_Dialect, place # "Context", traits> {
    let summary = !strconcat("!phi.context_", place, " type");
}

def PhiOpTrait : NativeOpTrait<"PhiOpTrait">;

def CPU_Allocator : AllocatorTypeOf<"CPU">;
def GPU_Allocator : AllocatorTypeOf<"GPU">;

def CPU_Context : ContextTypeOf<"CPU">;
def GPU_Context : ContextTypeOf<"GPU">;

def Allocator : AnyTypeOf<[CPU_Allocator, GPU_Allocator], "Allocator type">;
def Context : AnyTypeOf<[CPU_Context, GPU_Context], "Context type">;

#endif
