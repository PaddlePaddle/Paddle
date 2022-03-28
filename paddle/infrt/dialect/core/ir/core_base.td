#ifndef CORE_OPS_BASE
#define CORE_OPS_BASE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Core_Dialect : Dialect {
  let summary =
    "A dialect containing the Core Attributes, Operations, and Types";

  let name = "core";
  let cppNamespace = "::infrt::core";
  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;
}

// Type definitions
// Base class for Core dialect types.
class Core_Type<string name, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<Core_Dialect, name, traits, baseCppClass> {
}

class Core_EnumParam<string cppEnumType, string stringToSymbolFnName,
  string symbolToStringFnName, string desc = ""> : TypeParameter<cppEnumType, desc> {
  let parser = [{[&]() -> ::mlir::FailureOr<}] # cppEnumType # [{> {
    ::llvm::StringRef enumKeyword;
    if (::mlir::failed($_parser.parseKeyword(&enumKeyword)))
      return ::mlir::failure();
    auto maybeEnum = }] # stringToSymbolFnName # [{(enumKeyword);
    if (maybeEnum)
      return *maybeEnum;
    llvm_unreachable("}] # cppEnumType # [{ can not be found.");
    return {};
  }()}];
  let printer = "$_printer << " # symbolToStringFnName # "($_self)";
}

def TargetParam : Core_EnumParam<"TargetType", "GetTargetType", "GetString">;
def PrecisionParam : Core_EnumParam<"PrecisionType", "GetPrecisionType", "GetString">;
def LayoutParam : Core_EnumParam<"LayoutType", "GetLayoutType", "GetString">;

def TargetAttr : AttrDef<Core_Dialect, "Target"> {
  let mnemonic = "target";
  let parameters = (ins
    TargetParam:$target
  );
  let assemblyFormat = "`<` $target `>`";
}

def PrecisionAttr : AttrDef<Core_Dialect, "Precision"> {
  let mnemonic = "precision";
  let parameters = (ins
    PrecisionParam:$precision
  );
  let assemblyFormat = "`<` $precision `>`";
}

def LayoutAttr : AttrDef<Core_Dialect, "Layout"> {
  let mnemonic = "layout";
  let parameters = (ins
    LayoutParam:$layout
  );
  let assemblyFormat = "`<` $layout `>`";
}

def LoDTensor : Core_Type<"LoDTensor"> {
  let summary = "core lod tensor";
  let description = [{lod_tensor<3x64x3x3xf32, 3>}];
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType,
    "int32_t":$lod_level
  );
}

def DenseTensor : Core_Type<"DenseTensor"> {
  let summary = "core dense tensor";
  let description = [{dense_tensor<, 3>}];
  let parameters = (ins
    "::infrt::TargetType":$target,
    "::infrt::PrecisionType":$precision,
    "::infrt::LayoutType":$layout
  );
}
// Type Constrait for concrete DenseTensor type.
class DenseTensor<string target, string precision, string layout> :
    Type<CPred<"$_self == ::infrt::core::DenseTensorType::get($_self.getContext(), ::infrt::TargetType::"#target#",::infrt::PrecisionType::"#precision#",::infrt::LayoutType::"#layout#")">, 
    "!core.DenseTensor<"#target#","#precision#","#layout#">", 
    "::infrt::core::DenseTensorType">;

def DenseHostTensorMap :  Core_Type<"DenseHostTensorMap"> {
  let summary = "core dense tensor map";
  let description = [{dense_tensor map}];
  let parameters = (ins);
}

// TODO(wilber): Add !core.vec type.
def DenseTensorList :  Core_Type<"DenseTensorList"> {
  let summary = "core dense tensor map";
  let description = [{dense_tensor map}];
  let parameters = (ins);
}

// Base class for core dialect attributes.
class Core_Attr<string name, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<Core_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

// tools function. used for pattern rewriter
class CORE_createI32Attr<string value> : NativeCodeCall<
    "$_builder.getI32IntegerAttr(" # value # ")">;

class CORE_createSI32Attr<string value> : NativeCodeCall<
    "$_builder.getSI32IntegerAttr(" # value # ")">;

class CORE_createF32Attr<string value> : NativeCodeCall<
    "$_builder.getF32FloatAttr(" # value # ")">;

def CORE_cvtValueToValueRange : NativeCodeCall<
    "::infrt::cvtValueToValueRange($0)">;

def CORE_concatTwoValueRange : NativeCodeCall<
    "::infrt::concatTwoValueRange($0, $1)">;

#endif // CORE_OPS_BASE
