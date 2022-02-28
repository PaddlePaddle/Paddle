include "paddle/infrt/dialect/infrt/infrt_ops_base.td"

// Op definition
class Infrt_Op<string mnemonic, list<OpTrait> traits = []> : Op<Infrt_Dialect, mnemonic, traits> {

  // Each registered op needs to provide all of a printer, parser and verifier.
  // let printer = [{ return infrt::print(p, *this); }];
  // let verifier = [{ return infrt::verify(*this); }];
  // let parser = [{ return infrt::parse$cppClass(parser, result); }];
}

def Infrt_KernelOp : Infrt_Op<"kernel", [NoSideEffect]> {
  let summary = "kernel op";
  let description = [{kernel op!}];
  let arguments = (ins Variadic<AnyType>:$operands,
                       StrAttr:$name,
                       OptionalAttr<DictionaryAttr>:$attrs);
  let results = (outs Variadic<AnyType>);
}
