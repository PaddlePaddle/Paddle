#include "paddle/cinn/hlir/pe/broadcast.h"
#include "paddle/cinn/adt/op_equation_context.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/op_util.h"

CINN_REGISTER_HELPER(generate_shape_op) {
  CINN_REGISTER_OP(generate_shape)
      .describe("generate shape")
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible)
      .set_support_level(4);
}