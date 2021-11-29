#ifndef CINN_BASE
#define CINN_BASE

include "mlir/IR/OpBase.td"

def CINN_Dialect : Dialect {
  let name = "cinn";

  let description = [{
    The CINN host dialect.
  }];

  let cppNamespace = "::infrt::dialect";
}

// Type definitions
def StringType :
    Type<CPred<"$_self.isa<::infrt::dt::StringType>()">, "!cinn.string type">,
    BuildableType<"$_builder.getType<::infrt::dt::StringType>()">;

def TensorType :
    Type<CPred<"$_self.isa<::infrt::dt::TensorType>()">, "!cinn.tensor type">;

def TensorMapType :
    Type<CPred<"$_self.isa<::infrt::dt::TensorMapType>()">, "!cinn.tensor_map type">,
    BuildableType<"$_builder.getType<::infrt::dt::TensorMapType>()">;

def BufferType : OpaqueType<"b", "buffer", "buffer">;

class CINN_createI32Attr<string value> : NativeCodeCall<
    "mlir::createI32Attr($_builder, $_loc, " # value # ")">;

def CINN_cvtValueToValueRange : NativeCodeCall<
    "mlir::cvtValueToValueRange($0)">;

def CINN_concatTwoValueRange : NativeCodeCall<
    "mlir::concatTwoValueRange($0, $1)">;

class IsBoolAttrEq<string value> : Constraint<
    CPred<"($0.getValue() ==" # value # ")">,
    "Bool attrbute value constraint">;
#endif  // CINN_BASE
