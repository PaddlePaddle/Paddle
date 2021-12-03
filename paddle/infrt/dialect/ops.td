include "mlir/IR/OpBase.td"
include "paddle/infrt/dialect/infrt_base.td"


class INFRT_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<INFRT_Dialect, mnemonic, traits>;
