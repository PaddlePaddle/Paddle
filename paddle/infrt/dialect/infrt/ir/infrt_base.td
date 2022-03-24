#ifndef INFRT_OPS_BASE
#define INFRT_OPS_BASE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Infrt_Dialect : Dialect {
  let summary =
    "A dialect containing the Infrt Attributes, Operations, and Types";

  let name = "infrt";
  let cppNamespace = "::infrt";
  let useDefaultAttributePrinterParser = 1;
}

// Type definitions
// Base class for Infrt dialect types.
class Infrt_Type<string name, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<Infrt_Dialect, name, traits, baseCppClass> {
}

class Infrt_EnumParam<string cppEnumType, string stringToSymbolFnName,
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

def TargetParam : Infrt_EnumParam<"::infrt::TargetType", "GetTargetType", "GetString">;
def PrecisionParam : Infrt_EnumParam<"::infrt::PrecisionType", "GetPrecisionType", "GetString">;
def LayoutParam : Infrt_EnumParam<"::infrt::LayoutType", "GetLayoutType", "GetString">;

def TargetAttr : AttrDef<Infrt_Dialect, "Target"> {
  let mnemonic = "target";
  let parameters = (ins
    TargetParam:$target
  );
  let assemblyFormat = "`<` $target `>`";
}

def PrecisionAttr : AttrDef<Infrt_Dialect, "Precision"> {
  let mnemonic = "precision";
  let parameters = (ins
    PrecisionParam:$precision
  );
  let assemblyFormat = "`<` $precision `>`";
}

def LayoutAttr : AttrDef<Infrt_Dialect, "Layout"> {
  let mnemonic = "layout";
  let parameters = (ins
    LayoutParam:$layout
  );
  let assemblyFormat = "`<` $layout `>`";
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
    "::infrt::TargetType":$target,
    "::infrt::PrecisionType":$precision,
    "::infrt::LayoutType":$layout
  );
}

def DenseHostTensorMap :  Infrt_Type<"DenseHostTensorMap"> {
  let summary = "infrt dense tensor map";
  let description = [{dense_tensor map}];
  let parameters = (ins);
}

// TODO(wilber): Add !infrt.vec type.
def DenseTensorList :  Infrt_Type<"DenseTensorList"> {
  let summary = "infrt dense tensor map";
  let description = [{dense_tensor map}];
  let parameters = (ins);
}

// Type Constrait for concrete DenseTensor type.
class DenseTensor<string target, string precision, string layout> :
    Type<CPred<"$_self == ::infrt::DenseTensorType::get($_self.getContext(), ::infrt::TargetType::"#target#",::infrt::PrecisionType::"#precision#",::infrt::LayoutType::"#layout#")">, 
    "!infrt.DenseTensor<"#target#","#precision#","#layout#">", 
    "::infrt::DenseTensorType">;

// Base class for infrt dialect attributes.
class Infrt_Attr<string name, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<Infrt_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

// tools function. used for pattern rewriter
class INFRT_createI32Attr<string value> : NativeCodeCall<
    "$_builder.getI32IntegerAttr(" # value # ")">;

class INFRT_createSI32Attr<string value> : NativeCodeCall<
    "$_builder.getSI32IntegerAttr(" # value # ")">;

class INFRT_createF32Attr<string value> : NativeCodeCall<
    "$_builder.getF32FloatAttr(" # value # ")">;

def INFRT_cvtValueToValueRange : NativeCodeCall<
    "infrt::cvtValueToValueRange($0)">;

def INFRT_concatTwoValueRange : NativeCodeCall<
    "infrt::concatTwoValueRange($0, $1)">;

#endif // INFRT_OPS_BASE
