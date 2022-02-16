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

#include "paddle/fluid/framework/convert_utils.h"
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

  bool HasAttr(const std::string& name) const override {
    return ctx_.HasAttr(name);
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

  bool IsDenseTensorOutput(const std::string& name) const override {
    auto var_types = ctx_.GetOutputsVarType(name);
    return var_types[0] == proto::VarType::LOD_TENSOR;
  }

  bool IsSelectedRowsOutput(const std::string& name) const override {
    auto var_types = ctx_.GetOutputsVarType(name);
    return var_types[0] == proto::VarType::SELECTED_ROWS;
  }

 private:
  const InferShapeContext& ctx_;
};

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
      if (var->IsType<pten::DenseTensor>()) {
        return var->Get<pten::DenseTensor>().dims();
      } else if (var->IsType<pten::SelectedRows>()) {
        return var->Get<pten::SelectedRows>().dims();
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Currently, only can get dims from DenseTensor or SelectedRows."));
      }
    } else {
      auto* var = BOOST_GET_CONST(VarDesc*, var_);
      return make_ddim(var->GetShape());
    }
  }

  pten::DataType dtype() const override {
    if (is_runtime_) {
      auto* var = BOOST_GET_CONST(Variable*, var_);
      if (var->IsType<pten::DenseTensor>()) {
        return var->Get<pten::DenseTensor>().dtype();
      } else if (var->IsType<pten::SelectedRows>()) {
        return var->Get<pten::SelectedRows>().dtype();
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Currently, only can get dtype from DenseTensor or SelectedRows."));
      }
    } else {
      auto* var = BOOST_GET_CONST(VarDesc*, var_);
      return paddle::framework::TransToPtenDataType(var->GetDataType());
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
      if (var->IsType<pten::DenseTensor>()) {
        auto* tensor = var->GetMutable<pten::DenseTensor>();
        pten::DenseTensorUtils::GetMutableMeta(tensor)->dims = dims;
      } else if (var->IsType<pten::SelectedRows>()) {
        auto* tensor = var->GetMutable<pten::SelectedRows>()->mutable_value();
        pten::DenseTensorUtils::GetMutableMeta(tensor)->dims = dims;
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Currently, only can set dims from DenseTensor or SelectedRows."));
      }
    } else {
      auto* var = BOOST_GET(VarDesc*, var_);
      var->SetShape(vectorize(dims));
    }
  }

  void set_dtype(pten::DataType dtype) override {
    if (is_runtime_) {
      auto* var = BOOST_GET(Variable*, var_);
      if (var->IsType<pten::DenseTensor>()) {
        auto* tensor = var->GetMutable<pten::DenseTensor>();
        pten::DenseTensorUtils::GetMutableMeta(tensor)->dtype = dtype;
      } else if (var->IsType<pten::SelectedRows>()) {
        auto* tensor = var->GetMutable<pten::SelectedRows>()->mutable_value();
        pten::DenseTensorUtils::GetMutableMeta(tensor)->dtype = dtype;
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Currently, only can set dtype from DenseTensor or SelectedRows."));
      }
    } else {
      auto* var = BOOST_GET(VarDesc*, var_);
      var->SetDataType(paddle::framework::TransToProtoVarType(dtype));
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
      if (var->IsType<pten::DenseTensor>()) {
        auto* tensor = var->GetMutable<pten::DenseTensor>();
        pten::DenseTensorUtils::GetMutableMeta(tensor)->lod =
            static_cast<const CompatMetaTensor&>(meta_tensor).GetRuntimeLoD();
      } else {
        // NOTE(chenweihang): do nothing
        // only LoDTensor need to share lod
      }
    } else {
      auto* var = BOOST_GET(VarDesc*, var_);
      var->SetLoDLevel(static_cast<const CompatMetaTensor&>(meta_tensor)
                           .GetCompileTimeLoD());
    }
  }

  void share_meta(const MetaTensor& meta_tensor) override {
    set_dims(meta_tensor.dims());
    set_dtype(meta_tensor.dtype());
    // VarDesc doesn't contains layout, so we cannot share layout
    // set_layout(meta_tensor.layout());

    // special case 1: share lod of LoDTensor
    share_lod(meta_tensor);

    // special case 2: share height and rows of SelectedRows in runtime
    if (is_runtime_) {
      auto* var = BOOST_GET(Variable*, var_);
      if (var->IsType<pten::SelectedRows>()) {
        auto* selected_rows = var->GetMutable<pten::SelectedRows>();
        auto& input_selected_rows =
            static_cast<const CompatMetaTensor&>(meta_tensor).GetSelectedRows();
        selected_rows->set_rows(input_selected_rows.rows());
        selected_rows->set_height(input_selected_rows.height());
      }
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

  const pten::SelectedRows& GetSelectedRows() const {
    PADDLE_ENFORCE_EQ(is_runtime_, true,
                      platform::errors::Unavailable(
                          "Only can get Tensor from MetaTensor in rumtime."));
    auto* var = BOOST_GET_CONST(Variable*, var_);
    PADDLE_ENFORCE_EQ(var->IsType<pten::SelectedRows>(), true,
                      platform::errors::Unavailable(
                          "The Tensor in MetaTensor is not SelectedRows."));
    return var->Get<pten::SelectedRows>();
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
  auto& attr_names = std::get<1>(signature.args);
  auto& output_names = std::get<2>(signature.args);

  // TODO(chenweihang): support multiple inputs and outputs later
  pten::InferMetaContext infer_mete_context;
  for (auto& in_name : input_names) {
    if (ctx->HasInput(in_name)) {
      infer_meta_context.EmplaceBackInput(std::make_shared<CompatMetaTensor>(
          ctx->GetInputVarPtrs(in_name)[0], ctx->IsRuntime()));
    } else {
      infer_meta_context.EmplaceBackInput({nullptr});
    }
  }

  auto attr_reader = ctx->Attrs();
  for (auto& attr_name : attr_names) {
    if (ctx->HasAttr(attr_name)) {
      auto& attr = attr_reader.GetAttr(attr_name);
      if (std::type_index(attr.type()) == std::type_index(typeid(bool))) {
        infer_meta_context.EmplaceBackAttr(BOOST_GET_CONST(bool, attr));
      } else if (std::type_index(attr.type()) ==
                 std::type_index(typeid(float))) {
        infer_meta_context.EmplaceBackAttr(BOOST_GET_CONST(float, attr));
      } else {
        // do nothing, skip useless attrs now
        // TODO(chenweihang): support other attr type later and throw error
        // if attr is cannot parsed
      }
    } else {
      // do nothing
    }
  }

  for (auto& out_name : output_names) {
    if (ctx->HasOutput(out_name)) {
      infer_meta_context.EmplaceBackOutput(std::make_shared<CompatMetaTensor>(
          ctx->GetOutputVarPtrs(out_name)[0], ctx->IsRuntime()));
    } else {
      infer_meta_context.EmplaceBackOutput({nullptr});
    }
  }

  return infer_meta_context;
}

}  // namespace framework
}  // namespace paddle
