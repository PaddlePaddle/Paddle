#ifndef INFRT_OPS_BASE
#define INFRT_OPS_BASE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Infrt_Dialect : Dialect {
  let summary =
    "A dialect containing the Infrt Attributes, Operations, and Types";

  let name = "infrt";
  let cppNamespace = "::infrt";
}

// Type definitions

// Base class for Infrt dialect types.
class Infrt_Type<string name, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<Infrt_Dialect, name, traits, baseCppClass> {
}

def LoDTensor : Infrt_Type<"LoDTensor"> {
  let summary = "infrt lod tensor";
  let description = [{lod_tensor<3x64x3x3xf32, 3>}];
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType,
    "int32_t":$lod_level
  );
}

def DenseTensor : Infrt_Type<"DenseTensor"> {
  let summary = "infrt dense tensor";
  let description = [{dense_tensor<, 3>}];
  let parameters = (ins
    "TargetType":$target,
    "PrecisionType":$precision,
    "LayoutType":$layout
  );
}

// Base class for infrt dialect attributes.
class Infrt_Attr<string name, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<Infrt_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}
#endif // INFRT_OPS_BASE
