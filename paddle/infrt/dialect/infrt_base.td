#ifndef INFRT_BASE
#define INFRT_BASE

include "mlir/IR/OpBase.td"
include "paddle/infrt/dialect/infrt/infrt_ops_base.td"

def INFRT_Dialect : Dialect {
  let name = "Infrt";

  let description = [{
    The INFRT host dialect.
  }];

  let cppNamespace = "::infrt::dialect";
}

def BufferType : OpaqueType<"b", "buffer", "buffer">;

class INFRT_createI32Attr<string value> : NativeCodeCall<
    "infrt::createI32Attr($_builder, $_loc, " # value # ")">;

class INFRT_createSI32Attr<string value> : NativeCodeCall<
    "infrt::createSI32Attr($_builder, $_loc, " # value # ")">;

class INFRT_createF32Attr<string value> : NativeCodeCall<
    "infrt::createF32Attr($_builder, $_loc, " # value # ")">;

def INFRT_cvtValueToValueRange : NativeCodeCall<
    "infrt::cvtValueToValueRange($0)">;

def INFRT_concatTwoValueRange : NativeCodeCall<
    "infrt::concatTwoValueRange($0, $1)">;
#endif  // INFRT_BASE
