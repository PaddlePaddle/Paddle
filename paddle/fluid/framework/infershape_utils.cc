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

#include <string>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/pten_utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/scalar_array.h"
#include "paddle/phi/core/compat/arg_map_context.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/tensor_utils.h"

namespace paddle {
namespace framework {

class InferShapeArgumentMappingContext : public phi::ArgumentMappingContext {
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
    if (ctx_.HasInputs(name)) {
      return ctx_.Inputs(name).size();
    } else if (ctx_.HasInput(name)) {
      return 1;
    }
    return 0;
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
class CompatMetaTensor : public phi::MetaTensor {
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
      if (var->IsType<phi::DenseTensor>()) {
        return var->Get<phi::DenseTensor>().dims();
      } else if (var->IsType<phi::SelectedRows>()) {
        return var->Get<phi::SelectedRows>().dims();
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Currently, only can get dims from DenseTensor or SelectedRows."));
      }
    } else {
      auto* var = BOOST_GET_CONST(VarDesc*, var_);
      return phi::make_ddim(var->GetShape());
    }
  }

  phi::DataType dtype() const override {
    if (is_runtime_) {
      auto* var = BOOST_GET_CONST(Variable*, var_);
      if (var->IsType<phi::DenseTensor>()) {
        return var->Get<phi::DenseTensor>().dtype();
      } else if (var->IsType<phi::SelectedRows>()) {
        return var->Get<phi::SelectedRows>().dtype();
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
      if (var->IsType<phi::DenseTensor>()) {
        auto* tensor = var->GetMutable<phi::DenseTensor>();
        phi::DenseTensorUtils::GetMutableMeta(tensor)->dims = dims;
      } else if (var->IsType<phi::SelectedRows>()) {
        auto* tensor = var->GetMutable<phi::SelectedRows>()->mutable_value();
        phi::DenseTensorUtils::GetMutableMeta(tensor)->dims = dims;
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Currently, only can set dims from DenseTensor or SelectedRows."));
      }
    } else {
      auto* var = BOOST_GET(VarDesc*, var_);
      var->SetShape(vectorize(dims));
    }
  }

  void set_dtype(phi::DataType dtype) override {
    if (is_runtime_) {
      auto* var = BOOST_GET(Variable*, var_);
      if (var->IsType<phi::DenseTensor>()) {
        auto* tensor = var->GetMutable<phi::DenseTensor>();
        phi::DenseTensorUtils::GetMutableMeta(tensor)->dtype = dtype;
      } else if (var->IsType<phi::SelectedRows>()) {
        auto* tensor = var->GetMutable<phi::SelectedRows>()->mutable_value();
        phi::DenseTensorUtils::GetMutableMeta(tensor)->dtype = dtype;
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
      phi::DenseTensorUtils::GetMutableMeta(
          static_cast<phi::DenseTensor*>(tensor))
          ->layout = layout;
    } else {
      // NOTE(chenweihang): do nothing
      // Unsupported set layout for VarDesc now
    }
  }

  void share_lod(const MetaTensor& meta_tensor) override {
    if (is_runtime_) {
      auto* var = BOOST_GET(Variable*, var_);
      if (var->IsType<phi::DenseTensor>()) {
        auto* tensor = var->GetMutable<phi::DenseTensor>();
        phi::DenseTensorUtils::GetMutableMeta(tensor)->lod =
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
      if (var->IsType<phi::SelectedRows>()) {
        auto* selected_rows = var->GetMutable<phi::SelectedRows>();
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

  const phi::SelectedRows& GetSelectedRows() const {
    PADDLE_ENFORCE_EQ(is_runtime_, true,
                      platform::errors::Unavailable(
                          "Only can get Tensor from MetaTensor in rumtime."));
    auto* var = BOOST_GET_CONST(Variable*, var_);
    PADDLE_ENFORCE_EQ(var->IsType<phi::SelectedRows>(), true,
                      platform::errors::Unavailable(
                          "The Tensor in MetaTensor is not SelectedRows."));
    return var->Get<phi::SelectedRows>();
  }

  InferShapeVarPtr var_;
  bool is_runtime_;
};

phi::InferMetaContext BuildInferMetaContext(InferShapeContext* ctx,
                                            const std::string& op_type) {
  // 1. get kernel args
  InitDefaultKernelSignatureMap();
  auto arg_map_fn = phi::OpUtilsMap::Instance().GetArgumentMappingFn(op_type);
  PADDLE_ENFORCE_NOT_NULL(
      arg_map_fn, platform::errors::NotFound(
                      "The ArgumentMappingFn of %s op is not found.", op_type));
  InferShapeArgumentMappingContext arg_map_context(*ctx);
  auto signature = arg_map_fn(arg_map_context);
  VLOG(3) << "BuildInferMetaContext: op kernel signature - " << signature;

  // 2. build infermeta context
  phi::InferMetaContext infer_meta_context(ctx->IsRuntime());

  auto& input_names = std::get<0>(signature.args);
  auto& attr_names = std::get<1>(signature.args);
  auto& output_names = std::get<2>(signature.args);

  auto kernels_map =
      phi::KernelFactory::Instance().SelectKernelMap(signature.name);
  if (kernels_map.size() == 0) {
    PADDLE_THROW(
        platform::errors::Unimplemented("Not find `%s` kernels when construct "
                                        "InferMetaContext.",
                                        signature.name));
  }
  auto attr_defs = kernels_map.cbegin()->second.args_def().attribute_defs();

  // TODO(chenweihang): support multiple inputs and outputs later
  phi::InferMetaContext infer_mete_context;
  for (auto& in_name : input_names) {
    if (ctx->HasInput(in_name)) {
      infer_meta_context.EmplaceBackInput(std::make_shared<CompatMetaTensor>(
          ctx->GetInputVarPtrs(in_name)[0], ctx->IsRuntime()));
    } else {
      infer_meta_context.EmplaceBackInput({nullptr});
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
  auto attr_reader = ctx->Attrs();
  for (size_t i = 0; i < attr_names.size(); ++i) {
    auto attr_name = attr_names[i];
    if (attr_defs[i].type_index == std::type_index(typeid(phi::ScalarArray))) {
      // When attr is a vector_tensor or tensor, transform it to ScalarArray
      if (ctx->HasInputs(attr_name) || ctx->HasInput(attr_name)) {
        const auto& infershape_inputs = ctx->GetInputVarPtrs(attr_name);
        if (ctx->IsRuntime()) {
          // If is in runtime, we will get tensor's value for ScalarArray
          // and push it into attrs
          std::vector<Variable*> vars;
          vars.reserve(infershape_inputs.size());
          for (size_t i = 0; i < infershape_inputs.size(); i++) {
            vars.push_back(BOOST_GET_CONST(Variable*, infershape_inputs[i]));
          }
          if (infershape_inputs.size() != 1) {
            infer_meta_context.EmplaceBackAttr(
                std::move(experimental::MakePtenScalarArrayFromVarList(vars)));
          } else {
            infer_meta_context.EmplaceBackAttr(
                std::move(experimental::MakePtenScalarArrayFromVar(*vars[0])));
          }
        } else {
          // If is not in runtime, we will set default value(-1) for ScalarArray
          int64_t num_ele = 0;
          std::vector<VarDesc*> vars;
          vars.reserve(infershape_inputs.size());
          for (size_t i = 0; i < infershape_inputs.size(); i++) {
            vars.push_back(BOOST_GET_CONST(VarDesc*, infershape_inputs[i]));
          }

          if (vars.size() == 1) {
            num_ele = 1;
            const auto& tensor_dims = vars[0]->GetShape();
            for (size_t i = 0; i < tensor_dims.size(); ++i) {
              num_ele *= tensor_dims[i];
            }
          } else {
            for (auto& var : vars) {
              const auto& tensor_dims = var->GetShape();
              PADDLE_ENFORCE_EQ(tensor_dims.size(), 1,
                                platform::errors::InvalidArgument(
                                    "The shape is constructed by multi-tensor, "
                                    "every tensor's dims should be 1. But your "
                                    "shape has tensor that dims is %s.",
                                    tensor_dims.size()));
              num_ele += tensor_dims[0];
            }
          }
          phi::ScalarArray tensor_attr(std::vector<int32_t>(num_ele, -1));
          tensor_attr.SetFromTensor(true);
          infer_meta_context.EmplaceBackAttr(std::move(tensor_attr));
        }
      } else if (ctx->HasAttr(attr_name)) {
        auto& attr = attr_reader.GetAttr(attr_name);
        if (std::type_index(attr.type()) ==
            std::type_index(typeid(std::vector<int32_t>))) {
          infer_meta_context.EmplaceBackAttr(std::move(
              phi::ScalarArray(BOOST_GET_CONST(std::vector<int32_t>, attr))));
        } else {
          PADDLE_THROW(platform::errors::Unimplemented(
              "Unsupported cast op attribute `%s` to ScalarArray when "
              "construct KernelContext.",
              attr_name));
        }
      }
    } else if (attr_defs[i].type_index ==
               std::type_index(typeid(phi::Scalar))) {
      if (ctx->HasAttr(attr_name)) {
        // TODO(chentianyu03): support other attrs later
        auto& attr = attr_reader.GetAttr(attr_name);
        if (std::type_index(attr.type()) == std::type_index(typeid(float))) {
          infer_meta_context.EmplaceBackAttr(
              phi::Scalar(BOOST_GET_CONST(float, attr)));
        } else if (std::type_index(attr.type()) ==
                   std::type_index(typeid(std::string))) {
          infer_meta_context.EmplaceBackAttr(
              phi::Scalar(BOOST_GET_CONST(std::string, attr)));
        } else if (std::type_index(attr.type()) ==
                   std::type_index(typeid(int))) {
          infer_meta_context.EmplaceBackAttr(
              phi::Scalar(BOOST_GET_CONST(int, attr)));
        } else {
          PADDLE_THROW(platform::errors::Unimplemented(
              "Unsupported cast op attribute `%s` to Scalar when construct "
              "InferMetaContext.",
              attr_name));
        }
      } else if (ctx->HasInput(attr_name)) {
        const auto& infershape_input = ctx->GetInputVarPtrs(attr_name);

        if (infershape_input.size() == 1) {
          if (ctx->IsRuntime()) {
            Variable* var = BOOST_GET_CONST(Variable*, infershape_input[0]);
            infer_meta_context.EmplaceBackAttr(
                std::move(experimental::MakePtenScalarFromVar(*var)));
          } else {
            phi::Scalar tensor_scalar(-1);
            tensor_scalar.SetFromTensor(true);
            infer_meta_context.EmplaceBackAttr(std::move(tensor_scalar));
          }
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Invalid input.size() when cast op attribute `%s` to Scalar, "
              "expected 1, but actually is %d .",
              attr_name, infershape_input.size()));
        }
      }
    } else if (ctx->HasAttr(attr_name)) {
      // Emplace Back Attr according to the type of attr.
      auto& attr = attr_reader.GetAttr(attr_name);
      if (attr_defs[i].type_index == std::type_index(typeid(bool))) {
        infer_meta_context.EmplaceBackAttr(BOOST_GET_CONST(bool, attr));
      } else if (attr_defs[i].type_index == std::type_index(typeid(int))) {
        infer_meta_context.EmplaceBackAttr(BOOST_GET_CONST(int, attr));
      } else if (attr_defs[i].type_index == std::type_index(typeid(int64_t))) {
        infer_meta_context.EmplaceBackAttr(BOOST_GET_CONST(int64_t, attr));
      } else if (attr_defs[i].type_index == std::type_index(typeid(float))) {
        infer_meta_context.EmplaceBackAttr(BOOST_GET_CONST(float, attr));
      } else if (attr_defs[i].type_index ==
                 std::type_index(typeid(std::string))) {
        infer_meta_context.EmplaceBackAttr(BOOST_GET_CONST(std::string, attr));
      } else if (attr_defs[i].type_index ==
                 std::type_index(typeid(std::vector<bool>))) {
        infer_meta_context.EmplaceBackAttr(
            BOOST_GET_CONST(std::vector<bool>, attr));
      } else if (attr_defs[i].type_index ==
                 std::type_index(typeid(std::vector<int>))) {
        infer_meta_context.EmplaceBackAttr(
            BOOST_GET_CONST(std::vector<int>, attr));
      } else if (attr_defs[i].type_index ==
                 std::type_index(typeid(std::vector<int64_t>))) {
        if (std::type_index(attr.type()) ==
            std::type_index(typeid(std::vector<int>))) {
          // Emplace Back Attr according to the type of Phi_Kernel args.
          const auto& vector_int_attr = BOOST_GET_CONST(std::vector<int>, attr);
          const std::vector<int64_t> vector_int64_attr(vector_int_attr.begin(),
                                                       vector_int_attr.end());
          infer_meta_context.EmplaceBackAttr(vector_int64_attr);
        } else {
          infer_meta_context.EmplaceBackAttr(
              BOOST_GET_CONST(std::vector<int64_t>, attr));
        }
      } else if (attr_defs[i].type_index ==
                 std::type_index(typeid(std::vector<float>))) {
        infer_meta_context.EmplaceBackAttr(
            BOOST_GET_CONST(std::vector<float>, attr));
      } else if (attr_defs[i].type_index ==
                 std::type_index(typeid(std::vector<double>))) {
        infer_meta_context.EmplaceBackAttr(
            BOOST_GET_CONST(std::vector<double>, attr));
      } else if (attr_defs[i].type_index ==
                 std::type_index(typeid(std::vector<std::string>))) {
        infer_meta_context.EmplaceBackAttr(
            BOOST_GET_CONST(std::vector<std::string>, attr));
      } else if (attr_defs[i].type_index ==
                 std::type_index(typeid(phi::DataType))) {
        auto data_type = paddle::framework::TransToPtenDataType(
            static_cast<framework::proto::VarType::Type>(
                BOOST_GET_CONST(int, attr)));
        infer_meta_context.EmplaceBackAttr(data_type);
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported attribute type is received when call "
            "InferShapeFunctor."));
      }
    }
  }

  return infer_meta_context;
}

}  // namespace framework
}  // namespace paddle
