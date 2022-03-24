#include <mlir/IR/Builders.h>
#include <mlir/Transforms/DialectConversion.h>
#include "paddle/infrt/dialect/infrt/ir/infrt_dialect.h"
#include "paddle/infrt/dialect/tensorrt/trt_ops.h"

namespace infrt {
namespace trt {
static mlir::Value createTRTConv2dOp(mlir::PatternRewriter &rewriter, mlir::Operation *op) {
  ::mlir::Operation::operand_range Input(op->getOperands());
  ::mlir::Operation::operand_range Filter(op->getOperands());


  ::mlir::SmallVector<::mlir::Value, 4> operands;
  auto castedOp0 = ::llvm::dyn_cast<infrt::pd::Conv2dOp>(op); (void)castedOp0;
  Input = castedOp0.getODSOperands(0);
  Filter = castedOp0.getODSOperands(1);
  operands.push_back((*Input.begin()));
  operands.push_back((*Input.begin()));

  ::mlir::SmallVector<::mlir::Type, 4> resultTypes;
  for (auto v: castedOp0.getODSResults(0)) {
    resultTypes.push_back(v.getType());
  }
  ::mlir::SmallVector<::mlir::NamedAttribute, 8> attributes;
  {
    auto tblgen_attr = rewriter.getSI32IntegerAttr(3);
    attributes.emplace_back(rewriter.getStringAttr("out_channel_num"), tblgen_attr);
  }
  {
    auto tblgen_attr = rewriter.getI32ArrayAttr({3,3});
    attributes.emplace_back(rewriter.getStringAttr("kernel_size"), tblgen_attr);
  }
  {
    auto tblgen_attr = op->getAttrOfType<::mlir::ArrayAttr>("strides");(void)tblgen_attr;
    attributes.emplace_back(rewriter.getStringAttr("strides"), tblgen_attr);
  }
  {
    auto tblgen_attr = op->getAttrOfType<::mlir::ArrayAttr>("paddings");(void)tblgen_attr;
    attributes.emplace_back(rewriter.getStringAttr("paddings"), tblgen_attr);
  }
  {
    auto tblgen_attr = op->getAttrOfType<::mlir::StringAttr>("padding_algorithm");(void)tblgen_attr;
    attributes.emplace_back(rewriter.getStringAttr("padding_mode"), tblgen_attr);
  }
  {
    auto tblgen_attr = op->getAttrOfType<::mlir::IntegerAttr>("groups");(void)tblgen_attr;
    attributes.emplace_back(rewriter.getStringAttr("groups"), tblgen_attr);
  }
  {
    auto tblgen_attr = op->getAttrOfType<::mlir::ArrayAttr>("dilations");(void)tblgen_attr;
    attributes.emplace_back(rewriter.getStringAttr("dilations"), tblgen_attr);
  }
  {
    auto tblgen_attr = op->getAttrOfType<::mlir::StringAttr>("data_format");(void)tblgen_attr;
    attributes.emplace_back(rewriter.getStringAttr("data_format"), tblgen_attr);
  }
  return rewriter.create<trt::ConvolutionOp>(op->getLoc(), resultTypes, operands, attributes);
}
}  // namespace trt
}  // namespace infrt
