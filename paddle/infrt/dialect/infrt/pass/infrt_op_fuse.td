#ifndef INFRT_OP_FUSE
#define INFRT_OP_FUSE

include "mlir/Interfaces/SideEffectInterfaces.td"
include "paddle/infrt/dialect/infrt/infrt_ops.td"
include "paddle/infrt/dialect/pd_ops.td"

def FuseCvtTensorPattern : Pat<
       (Infrt_CvtTensorOp (Infrt_CvtTensorOp $arg)),
       (Infrt_CvtTensorOp $arg)>;

def FuseFeedCvtTensorPattern : Pat<
       (Infrt_CvtTensorOp (PD_FeedOp $name)),
       (PD_FeedOp $name)>;

def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantCvtTensorOptPattern : Pat<
  (Infrt_CvtTensorOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;



#endif // INFRT_OP_FUSE
