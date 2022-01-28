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

def AllocatorType :
    Type<CPred<"$_self.isa<::infrt::pten::AllocatorType>()">, "!pten.allocator type">,
    BuildableType<"$_builder.getType<::infrt::pten::AllocatorType>()">;

def ContextType :
    Type<CPred<"$_self.isa<::infrt::pten::ContextType>()">, "!pten.context type">,
    BuildableType<"$_builder.getType<::infrt::pten::ContextType>()">;

#endif
