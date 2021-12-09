#ifndef INFRT_BASE
#define INFRT_BASE

include "mlir/IR/OpBase.td"

def INFRT_Dialect : Dialect {
  let name = "infrt";

  let description = [{
    The INFRT host dialect.
  }];

  let cppNamespace = "::infrt::dialect";
}

// Type definitions
def StringType :
    Type<CPred<"$_self.isa<::infrt::dt::StringType>()">, "!infrt.string type">,
    BuildableType<"$_builder.getType<::infrt::dt::StringType>()">;

def TensorType :
    Type<CPred<"$_self.isa<::infrt::dt::TensorType>()">, "!infrt.tensor type">;

def TensorMapType :
    Type<CPred<"$_self.isa<::infrt::dt::TensorMapType>()">, "!infrt.tensor_map type">,
    BuildableType<"$_builder.getType<::infrt::dt::TensorMapType>()">;

def BufferType : OpaqueType<"b", "buffer", "buffer">;

class INFRT_createI32Attr<string value> : NativeCodeCall<
    "mlir::createI32Attr($_builder, $_loc, " # value # ")">;

def INFRT_cvtValueToValueRange : NativeCodeCall<
    "mlir::cvtValueToValueRange($0)">;

def INFRT_concatTwoValueRange : NativeCodeCall<
    "mlir::concatTwoValueRange($0, $1)">;

class IsBoolAttrEq<string value> : Constraint<
    CPred<"($0.getValue() ==" # value # ")">,
    "Bool attrbute value constraint">;
#endif  // INFRT_BASE
