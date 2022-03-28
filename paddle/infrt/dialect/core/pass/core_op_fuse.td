#ifndef CORE_OP_FUSE
#define CORE_OP_FUSE

include "mlir/Interfaces/SideEffectInterfaces.td"
include "paddle/infrt/dialect/core/ir/core_ops.td"
include "paddle/infrt/dialect/pd/ir/pd_ops.td"

def FuseTensorCastPattern : Pat<
       (Core_TensorCastOp (Core_TensorCastOp $arg)),
       (Core_TensorCastOp $arg)>;

def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantTensorCastOptPattern : Pat<
  (Core_TensorCastOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;



#endif // CORE_OP_FUSE
