#ifndef TD_PATTERNS
#define TD_PATTERNS

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Tosa/IR/TosaOps.td"

// // error: no matching function for call to _mlir::tosa::AddOp::build(mlir::OpBuilder&, mlir::OperationState&, mlir::Value&, mlir::Value&)_
// def : Pat<(Tosa_AddOp (Tosa_AddOp $x, (Tosa_ConstOp:$c1 $attr1)), (Tosa_ConstOp:$c2 $attr2)),
//  (Tosa_AddOp $x, (Tosa_AddOp:$const_val $c1, $c2))>;

def UpdateAddOp : NativeCodeCall<"UpdateAddOp($_builder, $_loc, $0, $1, $2)">;
def : Pat<(Tosa_AddOp (Tosa_AddOp $x, (Tosa_ConstOp:$c1 $attr1)), (Tosa_ConstOp:$c2 $attr2)),
  (UpdateAddOp $x, $c1, $c2)>;


#endif  // TD_PATTERNS
