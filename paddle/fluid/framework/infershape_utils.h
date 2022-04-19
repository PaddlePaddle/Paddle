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

#pragma once

#include <string>

#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"

namespace paddle {
namespace framework {

// TODO(chenweihang): Support TensorArray later
class CompatMetaTensor : public phi::MetaTensor {
 public:
  explicit CompatMetaTensor(bool is_runtime)
      : is_runtime_(is_runtime), initialized_(false) {}
  CompatMetaTensor(InferShapeVarPtr var, bool is_runtime)
      : var_(std::move(var)), is_runtime_(is_runtime) {}

  CompatMetaTensor(CompatMetaTensor&&) = default;
  CompatMetaTensor& operator=(CompatMetaTensor&&) = default;
  CompatMetaTensor(const CompatMetaTensor&) = default;
  CompatMetaTensor& operator=(const CompatMetaTensor&) = default;

  int64_t numel() const override;

  DDim dims() const override;

  phi::DataType dtype() const override;

  DataLayout layout() const override;

  void set_dims(const DDim& dims) override;

  void set_dtype(phi::DataType dtype) override;

  void set_layout(DataLayout layout) override;

  void share_lod(const MetaTensor& meta_tensor) override;

  void share_dims(const MetaTensor& meta_tensor) override;

  void share_meta(const MetaTensor& meta_tensor) override;

  bool initialized() const override { return initialized_; };

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
  bool initialized_{true};
};

// Note: In order to avoid using shared_ptr to manage MetaTensor in
// InferMetaContext, inherit and implement InferMetaContext separately
// for compatibility with fluid, shared_ptr will cause significant decrease
// in scheduling performance
class CompatInferMetaContext : public phi::InferMetaContext {
 public:
  CompatInferMetaContext() = default;
  explicit CompatInferMetaContext(phi::MetaConfig config)
      : phi::InferMetaContext(config) {}

  void EmplaceBackInput(CompatMetaTensor input);
  void EmplaceBackOutput(CompatMetaTensor output);

  void EmplaceBackInputs(
      paddle::SmallVector<CompatMetaTensor, phi::kInputSmallVectorSize> inputs);
  void EmplaceBackOutputs(
      paddle::SmallVector<CompatMetaTensor, phi::kOutputSmallVectorSize>
          outputs);

  const phi::MetaTensor& InputAt(size_t idx) const override;
  paddle::optional<const phi::MetaTensor&> OptionalInputAt(
      size_t idx) const override;

  std::vector<const phi::MetaTensor*> InputsBetween(size_t start,
                                                    size_t end) const override;
  paddle::optional<const std::vector<const phi::MetaTensor*>>
  OptionalInputsBetween(size_t start, size_t end) const override;

  phi::MetaTensor* MutableOutputAt(size_t idx) override;
  std::vector<phi::MetaTensor*> MutableOutputBetween(size_t start,
                                                     size_t end) override;

  virtual ~CompatInferMetaContext() = default;

 private:
  paddle::SmallVector<CompatMetaTensor, phi::kInputSmallVectorSize>
      compat_inputs_;
  paddle::SmallVector<CompatMetaTensor, phi::kOutputSmallVectorSize>
      compat_outputs_;
};

CompatInferMetaContext BuildInferMetaContext(InferShapeContext* ctx,
                                             const std::string& op_type);

#define DECLARE_INFER_SHAPE_FUNCTOR(op_type, functor_name, fn)      \
  struct functor_name : public paddle::framework::InferShapeBase {  \
    void operator()(                                                \
        paddle::framework::InferShapeContext* ctx) const override { \
      auto infer_meta_context =                                     \
          paddle::framework::BuildInferMetaContext(ctx, #op_type);  \
      fn(&infer_meta_context);                                      \
    }                                                               \
  }

}  // namespace framework
}  // namespace paddle
