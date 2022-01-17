#include "paddle/infrt/dialect/pd_test_op_kernel_mapping_passes.h"

#include <mlir/Pass/Pass.h>
#include <glog/logging.h>

#include "paddle/infrt/dialect/pd_ops.h"
#include "paddle/infrt/dialect/pd_test_op_kernel_mappings.h"

namespace infrt {

struct OpKernelMapPass : public mlir::PassWrapper<OpKernelMapPass, mlir::OperationPass<>> {
 public:
  void runOnOperation() override {
    mlir::Operation* op = getOperation();
    op->walk([&](mlir::Operation* op) {
      LOG(INFO) << "Op.name: " << op->getName().getIdentifier().data();
      if (op->getName().getIdentifier().str() == "PD_MatmulOp") {
        pd::PDKEL_Matmul_to_CPU pattern(&getContext());
        pattern.matchAndRewrite(*op);
      }
    });
  }

};

}  // namespace infrt
