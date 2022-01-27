/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/infershape_utils.h"

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/pten_utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/pten/core/compat/arg_map_context.h"
#include "paddle/pten/core/compat/convert_utils.h"
#include "paddle/pten/core/compat/op_utils.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/infermeta_utils.h"
#include "paddle/pten/core/meta_tensor.h"
#include "paddle/pten/core/tensor_utils.h"

namespace paddle {
namespace framework {

class InferShapeArgumentMappingContext : public pten::ArgumentMappingContext {
 public:
  explicit InferShapeArgumentMappingContext(const InferShapeContext& ctx)
      : ctx_(ctx) {}

  bool HasInput(const std::string& name) const override {
    return ctx_.HasInput(name);
  }

  bool HasOutput(const std::string& name) const override {
    return ctx_.HasOutput(name);
  }

  paddle::any Attr(const std::string& name) const override {
    auto& attr = ctx_.Attrs().GetAttr(name);
    return GetAttrValue(attr);
  }

  size_t InputSize(const std::string& name) const override {
    return ctx_.Inputs(name).size();
  }

  size_t OutputSize(const std::string& name) const override {
    return ctx_.Outputs(name).size();
  }

  bool IsDenseTensorInput(const std::string& name) const override {
    auto var_types = ctx_.GetInputsVarType(name);
    return var_types[0] == proto::VarType::LOD_TENSOR;
  }

  bool IsSelectedRowsInput(const std::string& name) const override {
    auto var_types = ctx_.GetInputsVarType(name);
    return var_types[0] == proto::VarType::SELECTED_ROWS;
  }

 private:
  const InferShapeContext& ctx_;
};

// TODO(chenweihang): Support SelectedRows later
// TODO(chenweihang): Support TensorArray later
class CompatMetaTensor : public pten::MetaTensor {
 public:
  CompatMetaTensor(InferShapeVarPtr var, bool is_runtime)
      : var_(std::move(var)), is_runtime_(is_runtime) {}

  CompatMetaTensor() = default;
  CompatMetaTensor(const CompatMetaTensor&) = default;
  CompatMetaTensor(CompatMetaTensor&&) = default;
  CompatMetaTensor& operator=(const CompatMetaTensor&) = delete;
  CompatMetaTensor& operator=(CompatMetaTensor&&) = delete;

  int64_t numel() const override {
    if (is_runtime_) {
      auto* var = BOOST_GET_CONST(Variable*, var_);
      return var->Get<Tensor>().numel();
    } else {
      auto* var = BOOST_GET_CONST(VarDesc*, var_);
      return var->ElementSize();
    }
  }

  DDim dims() const override {
    if (is_runtime_) {
      auto* var = BOOST_GET_CONST(Variable*, var_);
      return var->Get<LoDTensor>().dims();
    } else {
      auto* var = BOOST_GET_CONST(VarDesc*, var_);
      return make_ddim(var->GetShape());
    }
  }

  pten::DataType dtype() const override {
    if (is_runtime_) {
      auto* var = BOOST_GET_CONST(Variable*, var_);
      return var->Get<LoDTensor>().dtype();
    } else {
      auto* var = BOOST_GET_CONST(VarDesc*, var_);
      return pten::TransToPtenDataType(var->GetDataType());
    }
  }

  DataLayout layout() const override {
    if (is_runtime_) {
      auto* var = BOOST_GET_CONST(Variable*, var_);
      return var->Get<LoDTensor>().layout();
    } else {
      // NOTE(chenweihang): do nothing
      // Unsupported get layout for VarDesc now
      return DataLayout::UNDEFINED;
    }
  }

  void set_dims(const DDim& dims) override {
    if (is_runtime_) {
      auto* var = BOOST_GET(Variable*, var_);
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      pten::DenseTensorUtils::GetMutableMeta(
          static_cast<pten::DenseTensor*>(tensor))
          ->dims = dims;
    } else {
      auto* var = BOOST_GET(VarDesc*, var_);
      var->SetShape(vectorize(dims));
    }
  }

  void set_dtype(pten::DataType dtype) override {
    if (is_runtime_) {
      auto* var = BOOST_GET(Variable*, var_);
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      pten::DenseTensorUtils::GetMutableMeta(
          static_cast<pten::DenseTensor*>(tensor))
          ->dtype = dtype;
    } else {
      auto* var = BOOST_GET(VarDesc*, var_);
      var->SetDataType(pten::TransToProtoVarType(dtype));
    }
  }

  void set_layout(DataLayout layout) override {
    if (is_runtime_) {
      auto* var = BOOST_GET(Variable*, var_);
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      pten::DenseTensorUtils::GetMutableMeta(
          static_cast<pten::DenseTensor*>(tensor))
          ->layout = layout;
    } else {
      // NOTE(chenweihang): do nothing
      // Unsupported set layout for VarDesc now
    }
  }

  void share_lod(const MetaTensor& meta_tensor) override {
    if (is_runtime_) {
      auto* var = BOOST_GET(Variable*, var_);
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      pten::DenseTensorUtils::GetMutableMeta(
          static_cast<pten::DenseTensor*>(tensor))
          ->lod =
          static_cast<const CompatMetaTensor&>(meta_tensor).GetRuntimeLoD();
    } else {
      auto* var = BOOST_GET(VarDesc*, var_);
      var->SetLoDLevel(static_cast<const CompatMetaTensor&>(meta_tensor)
                           .GetCompileTimeLoD());
    }
  }

 private:
  const LoD& GetRuntimeLoD() const {
    auto* var = BOOST_GET_CONST(Variable*, var_);
    return var->Get<LoDTensor>().lod();
  }
  int32_t GetCompileTimeLoD() const {
    auto* var = BOOST_GET_CONST(VarDesc*, var_);
    return var->GetLoDLevel();
  }

  InferShapeVarPtr var_;
  bool is_runtime_;
};

pten::InferMetaContext BuildInferMetaContext(InferShapeContext* ctx,
                                             const std::string& op_type) {
  // 1. get kernel args
  InitDefaultKernelSignatureMap();
  auto arg_map_fn = pten::OpUtilsMap::Instance().GetArgumentMappingFn(op_type);
  PADDLE_ENFORCE_NOT_NULL(
      arg_map_fn, platform::errors::NotFound(
                      "The ArgumentMappingFn of %s op is not found.", op_type));
  InferShapeArgumentMappingContext arg_map_context(*ctx);
  auto signature = arg_map_fn(arg_map_context);
  VLOG(3) << "BuildInferMetaContext: op kernel signature - " << signature;

  // 2. build infermeta context
  pten::InferMetaContext infer_meta_context(ctx->IsRuntime());

  auto& input_names = std::get<0>(signature.args);
  auto& output_names = std::get<2>(signature.args);
  // TODO(chenweihang): support attrs in next pr
  // auto& attr_names = std::get<1>(signature.args);

  // TODO(chenweihang): support multiple inputs and outputs
  pten::InferMetaContext infer_mete_context;
  for (auto& in_name : input_names) {
    infer_meta_context.EmplaceBackInput(std::make_shared<CompatMetaTensor>(
        ctx->GetInputVarPtrs(in_name)[0], ctx->IsRuntime()));
  }
  for (auto& out_name : output_names) {
    infer_meta_context.EmplaceBackOutput(std::make_shared<CompatMetaTensor>(
        ctx->GetOutputVarPtrs(out_name)[0], ctx->IsRuntime()));
  }
  // TODO(chenweihang): support attrs later

  return infer_meta_context;
}

}  // namespace framework
}  // namespace paddle
