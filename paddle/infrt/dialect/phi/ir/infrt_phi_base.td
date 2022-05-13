#ifndef PHI_BASE
#define PHI_BASE

include "mlir/IR/OpBase.td"
include "paddle/infrt/dialect/infrt/ir/infrt_base.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def PHI_Dialect : Dialect {
  let name = "phi";

  let description = [{
    The PHI host dialect.
  }];

  let cppNamespace = "::infrt::phi";
  let useDefaultTypePrinterParser = 1;
}

def PhiOpTrait : NativeOpTrait<"PhiOpTrait">;

class PHI_Type<string type, list<Trait> traits = [], string baseCppClass = "::mlir::Type">
   : TypeDef<PHI_Dialect, type, !listconcat(traits, [PhiOpTrait, IsolatedFromAbove]), baseCppClass> {}

def Allocator : PHI_Type<"Allocator"> {
   let mnemonic = "allocator";
   let parameters = (ins
     TargetParam:$target
   );
   let assemblyFormat = "`<` $target `>`";
 }

 def Context : PHI_Type<"Context"> {
   let mnemonic = "context";
   let parameters = (ins
     TargetParam:$target
   );
   let assemblyFormat = "`<` $target `>`";
 }

def PD_DenseTensorMap : PHI_Type<"DenseTensorMap"> {
  let mnemonic = "dense_tensor_map";
}

#endif
