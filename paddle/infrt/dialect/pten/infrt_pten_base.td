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
    TypeDef<PTEN_Dialect, place # "AllocatorType", traits> {
    let summary = !strconcat("!pten.allocator_", place, " type");
}

class ContextTypeOf<string place, list<Trait> traits=[]>:
    TypeDef<PTEN_Dialect, place # "ContextType", traits> {
    let summary = !strconcat("!pten.context_", place, " type");
}

def CPU_AllocatorType : AllocatorTypeOf<"CPU">;
def GPU_AllocatorType : AllocatorTypeOf<"GPU">;

def CPU_ContextType : ContextTypeOf<"CPU">;
def GPU_ContextType : ContextTypeOf<"GPU">;

def AllocatorType : AnyTypeOf<[CPU_AllocatorType, GPU_AllocatorType], "Allocator type">;
def ContextType : AnyTypeOf<[CPU_ContextType, GPU_ContextType], "Context type">;

#endif
