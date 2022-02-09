#ifndef PTEN_BASE
#define PTEN_BASE

include "mlir/IR/OpBase.td"

def PTEN_Dialect : Dialect {
  let name = "pten";

  let description = [{
    The PTEN host dialect.
  }];

  let cppNamespace = "::infrt::pten";
}

class AllocatorTypeOf<string place, list<Trait> traits=[]>:
    TypeDef<PTEN_Dialect, place # "Allocator", traits> {
    let summary = !strconcat("!pten.allocator_", place, " type");
}

class ContextTypeOf<string place, list<Trait> traits=[]>:
    TypeDef<PTEN_Dialect, place # "Context", traits> {
    let summary = !strconcat("!pten.context_", place, " type");
}

def CPU_Allocator : AllocatorTypeOf<"CPU">;
def GPU_Allocator : AllocatorTypeOf<"GPU">;

def CPU_Context : ContextTypeOf<"CPU">;
def GPU_Context : ContextTypeOf<"GPU">;

def Allocator : AnyTypeOf<[CPU_Allocator, GPU_Allocator], "Allocator type">;
def Context : AnyTypeOf<[CPU_Context, GPU_Context], "Context type">;

#endif
