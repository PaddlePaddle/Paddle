#ifndef INFRT_OP_FUSE
#define INFRT_OP_FUSE

include "mlir/Interfaces/SideEffectInterfaces.td"
include "paddle/infrt/dialect/infrt/ir/infrt_ops.td"
include "paddle/infrt/dialect/pd/ir/pd_ops.td"

def FuseTensorCastPattern : Pat<
       (Infrt_TensorCastOp (Infrt_TensorCastOp $arg)),
       (Infrt_TensorCastOp $arg)>;

def FuseFeedTensorCastPattern : Pat<
       (Infrt_TensorCastOp (PD_FeedOp $name)),
       (PD_FeedOp $name)>;

def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantTensorCastOptPattern : Pat<
  (Infrt_TensorCastOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;



#endif // INFRT_OP_FUSE
