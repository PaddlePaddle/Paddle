#ifndef PHI_BASE
#define PHI_BASE

include "mlir/IR/OpBase.td"

def PHI_Dialect : Dialect {
  let name = "phi";

  let description = [{
    The PHI host dialect.
  }];

  let cppNamespace = "::infrt::phi";
}

def PhiOpTrait : NativeOpTrait<"PhiOpTrait">;

class PHI_Type<string type, list<Trait> traits = []>
  : TypeDef<PHI_Dialect, type, !listconcat(traits, [PhiOpTrait, IsolatedFromAbove])> {}

class PHI_TypeParameter<string cppEnumType, string stringToSymbolFnName,
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

def Target : PHI_TypeParameter<"::infrt::TargetType", "GetTargetType", "GetString">;

def Allocator : PHI_Type<"Allocator"> {
  let mnemonic = "allocator";
  let parameters = (ins
    Target:$target
  );
  let assemblyFormat = "`<` $target `>`";
}

def Context : PHI_Type<"Context"> {
  let mnemonic = "context";
  let parameters = (ins
    Target:$target
  );
  let assemblyFormat = "`<` $target `>`";
}

#endif
