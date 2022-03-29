// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/infrt/dialect/infrt/pass/infrt_weights_unfold_pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "paddle/infrt/dialect/infrt/common/types.h"
#include "paddle/infrt/dialect/infrt/ir/infrt_dialect.h"
#include "paddle/infrt/dialect/phi/ir/infrt_phi_tensor.h"
#include "paddle/infrt/dialect/phi/ir/phi_base.h"
#include "paddle/infrt/paddle/model_parser.h"
#include "paddle/infrt/tensor/phi/tensor_map.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/dense_tensor.h"

namespace paddle {
namespace platform {
using DeviceContext = ::phi::DeviceContext;
}  // namespace platform
namespace framework {
using LoDTensor = ::phi::DenseTensor;
void DeserializeFromStream(std::istream& is,
                           LoDTensor* tensor,
                           const platform::DeviceContext& dev_ctx);
}  // namespace framework
}  // namespace paddle

namespace {

class InfrtWeightsFoldPass
    : public mlir::PassWrapper<InfrtWeightsFoldPass, mlir::FunctionPass> {
 public:
  ::llvm::StringRef getName() const override { return "InfrtWeightsFoldPass"; }

  void runOnFunction() override;
};

static ::infrt::phi::DenseTensorMap LoadCombinedParams(
    llvm::StringRef model_path, llvm::StringRef params_path) {
  ::infrt::phi::DenseTensorMap map;

  auto pb_proto_prog = ::infrt::paddle::LoadProgram(model_path.str());
  auto main_block = pb_proto_prog->blocks(0);

  std::ifstream param_file(params_path.str(), std::ios::binary);

  std::set<std::string> tmp;
  for (auto& var : main_block.vars()) {
    if (var.name() == "feed" || var.name() == "fetch" || !var.persistable()) {
      continue;
    }
    if (var.type().type() ==
        ::paddle::framework::proto::VarType_Type_LOD_TENSOR) {
      tmp.emplace(var.name());
    } else {
      llvm_unreachable("the tensor type is illegal.");
    }
  }

  for (auto& var : tmp) {
    std::unique_ptr<::phi::DenseTensor> tensor{
        std::make_unique<::phi::DenseTensor>()};
    ::phi::CPUContext ctx;
    ::paddle::framework::DeserializeFromStream(param_file, tensor.get(), ctx);
    map.SetDenseTensor(var, std::move(tensor));
  }

  return map;
}

void InfrtWeightsFoldPass::runOnFunction() {
  mlir::Block& block = getFunction().body().front();
  mlir::OpBuilder builder(&block, block.begin());

  ::llvm::StringRef model_path, params_path;
  std::vector<mlir::Operation*> delete_op_list;
  // Insert cpu context. If the pass failed, the context op will be removed by
  // CanonicalizerPass.
  auto context_op = builder.create<infrt::phi::CreateCPUContextOp>(
      block.front().getLoc(),
      infrt::phi::ContextType::get(builder.getContext(),
                                   infrt::TargetType::CPU));

  for (auto& org_op : block) {
    if (auto op = llvm::dyn_cast<::infrt::phi::LoadCombinedParamsOp>(org_op)) {
      model_path = op.model_path();
      params_path = op.params_path();

      // Load params.
      auto map = LoadCombinedParams(model_path, params_path);
      bool delete_load_combined_op{false};
      // Find all use of map.
      for (auto map_arg : op.getODSResults(0)) {
        for (mlir::Operation* user_op : map_arg.getUsers()) {
          if (auto tensor_map_get_op =
                  llvm::dyn_cast<::infrt::phi::TensorMapGetTensorOp>(user_op)) {
            ::llvm::StringRef arg_name = tensor_map_get_op.name();
            ::phi::DenseTensor* tensor = map.GetDenseTensor(arg_name.str());

            builder.setInsertionPoint(tensor_map_get_op);
            auto inited_weight_op =
                builder.create<::infrt::phi::CreateInitedDenseTensorOp>(
                    tensor_map_get_op.getLoc(),
                    tensor_map_get_op.output().getType(),
                    context_op.output(),
                    builder.getI64ArrayAttr(
                        {tensor->dims().Get(),
                         tensor->dims().Get() + tensor->dims().size()}),
                    ::infrt::LayoutAttr::get(builder.getContext(),
                                             ::infrt::LayoutType::NCHW),
                    builder.getI64ArrayAttr({}),
                    builder.getF32ArrayAttr(
                        {tensor->data<float>(),
                         static_cast<size_t>(tensor->numel())}));
            tensor_map_get_op.replaceAllUsesWith(inited_weight_op.output());
            delete_load_combined_op = true;
            delete_op_list.push_back(tensor_map_get_op);
          }
        }
      }
      if (delete_load_combined_op) {
        delete_op_list.push_back(op);
      }
    }
  }

  // remove all map releation op.
  for (size_t i = 0; i < delete_op_list.size(); ++i) {
    delete_op_list[i]->erase();
  }
}

}  // namespace

std::unique_ptr<mlir::Pass> infrt::createInfrtWeightsUnfoldPass() {
  return std::make_unique<InfrtWeightsFoldPass>();
}
