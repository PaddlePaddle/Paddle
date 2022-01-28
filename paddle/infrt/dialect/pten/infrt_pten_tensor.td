#ifdef PTEN_TENSOR
#else
#define PTEN_TENSOR

include "paddle/infrt/dialect/pten/pten_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def PTEN_DenseTensorDialect : Dialect {
  let name = "pten_dt";

  let description = [{
    The PTEN DenseTensor dialect.
  }];

  let cppNamespace = "::infrt::dialect";
}


// PTEN DenseTensor related Op.
class PTEN_DT_Op<string mnemonic, list<OpTrait> traits = []> : Op<PTEN_DenseTensorDialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {
  // Each registered op needs to provide all of a printer, parser and verifier.
  let printer = [{ return infrt::dialect::print(p, *this); }];
  let verifier = [{ return infrt::dialect::verify(*this); }];
  let parser = [{ return infrt::dialect::parse$cppClass(parser, result); }];
}

#endif
