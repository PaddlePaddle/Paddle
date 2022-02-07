#ifndef PTEN_BASE
#define PTEN_BASE

include "mlir/IR/OpBase.td"

def PTEN_Dialect : Dialect {
  let name = "pten";

  let description = [{
    The PTEN host dialect.
  }];

  let cppNamespace = "::infrt::dialect";
}


def PTEN_DenseTensorDialect : Dialect {
  let name = "pten_dt";

  let description = [{
    The PTEN DenseTensor dialect.
  }];

  let cppNamespace = "::infrt::dialect";
}


class AllocatorType<string place, list<Trait> traits=[]>:
    TypeDef<PTEN_Dialect, place, traits> {
    let summary = !strconcat("!pten.allocator_", place, " type");
}

class ContextType<string place, list<Trait> traits=[]>:
    TypeDef<PTEN_Dialect, place, traits> {
    let summary = !strconcat("!pten.context_", place, " type");
}

def CPU_AllocatorType : AllocatorType<"CPU">;
def GPU_AllocatorType : AllocatorType<"GPU">;

def CPU_ContextType : ContextType<"CPU">;
def GPU_ContextType : ContextType<"GPU">;

def AllocatorType : AnyTypeOf<[CPU_AllocatorType, GPU_AllocatorType], "Allocator type">;
def ContextType : AnyTypeOf<[CPU_ContextType, GPU_ContextType], "Context type">;

#endif
