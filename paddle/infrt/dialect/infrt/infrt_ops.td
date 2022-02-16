#ifndef Infrt_OpS
#define Infrt_OpS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Infrt_Dialect : Dialect {
  let summary =
    "A dialect containing the Infrt Attributes, Operations, and Types";

  let name = "Infrt";
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

// Op definition
class Infrt_Op<string mnemonic, list<OpTrait> traits = []> : Op<Infrt_Dialect, mnemonic, traits> {

  // Each registered op needs to provide all of a printer, parser and verifier.
  // let printer = [{ return infrt::print(p, *this); }];
  // let verifier = [{ return infrt::verify(*this); }];
  // let parser = [{ return infrt::parse$cppClass(parser, result); }];
}

// def InfRT_KernelOp : Infrt_Op<"kernel", [NoSideEffect]> {
//  let summary = "kernel op";
//  let description = [{
//    kernel op!
//  }];
// let arguments = (ins StrAttr:$name, PD_Tensor:$X, PD_Tensor:$Y, DefaultValuedAttr<F32Attr, "1.0">:$Alpha, DefaultValuedAttr<F32Attr, "1.0">:$Beta);
//
// let results = (outs PD_Tensor:$Out);
// }

#endif // Infrt_OpS
