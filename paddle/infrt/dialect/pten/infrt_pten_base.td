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

def HostAllocator : AllocatorTypeOf<"Host">;
def GpuAllocator : AllocatorTypeOf<"Gpu">;

def HostContext : ContextTypeOf<"Host">;
def GpuContext : ContextTypeOf<"Gpu">;

def Allocator : AnyTypeOf<[HostAllocator, GpuAllocator], "Allocator type">;
def Context : AnyTypeOf<[HostContext, GpuContext], "Context type">;

#endif
