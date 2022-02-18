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

class DTTensorType<string target, string layout, string precision> :
    Type<CPred<"$_self.isa<::infrt::dt::TensorType>()">, "!infrt.tensor type", "::infrt::dt::TensorType">,
    BuildableType<"_builder.getType<::infrt::dt::TensorType,::infrt::dt::TargetType,::infrt::dt::LayoutType, ::infrt::dt::PrecisionType >(" #"::infrt::dt::"#target#","#"::infrt::dt::"#layout#","#"::infrt::dt::"#precision#")" >;


def TensorType :
    Type<CPred<"$_self.isa<::infrt::dt::TensorType>()">, "!infrt.tensor type", "::infrt::dt::TensorType">;


def TensorMapType :
    Type<CPred<"$_self.isa<::infrt::dt::TensorMapType>()">, "!infrt.tensor_map type">,
    BuildableType<"$_builder.getType<::infrt::dt::TensorMapType>()">;

def BufferType : OpaqueType<"b", "buffer", "buffer">;

class INFRT_createI32Attr<string value> : NativeCodeCall<
    "infrt::createI32Attr($_builder, $_loc, " # value # ")">;

def INFRT_cvtValueToValueRange : NativeCodeCall<
    "infrt::cvtValueToValueRange($0)">;

def INFRT_concatTwoValueRange : NativeCodeCall<
    "infrt::concatTwoValueRange($0, $1)">;
#endif  // INFRT_BASE
