// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include "paddle/fluid/eager/api/manual/eager_manual/nodes/nodes.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/tensor_wrapper.h"

class AcosGradNode : public egr::GradNodeBase {
 public:
  AcosGradNode() : egr::GradNodeBase() {}
  AcosGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AcosGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AcosGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AcosGradNode>(new AcosGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class AcoshGradNode : public egr::GradNodeBase {
 public:
  AcoshGradNode() : egr::GradNodeBase() {}
  AcoshGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AcoshGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AcoshGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AcoshGradNode>(new AcoshGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class AddmmGradNode : public egr::GradNodeBase {
 public:
  AddmmGradNode() : egr::GradNodeBase() {}
  AddmmGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AddmmGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AddmmGradNode"; }

  void ClearTensorWrappers() override {
    input_.clear();
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AddmmGradNode>(new AddmmGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperinput(const paddle::experimental::Tensor& input) {
    input_ = egr::TensorWrapper(input, false);
  }
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes
  void SetAttributealpha(const float& alpha) { alpha_ = alpha; }
  void SetAttributebeta(const float& beta) { beta_ = beta; }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
  float alpha_;
  float beta_;
};

class AngleGradNode : public egr::GradNodeBase {
 public:
  AngleGradNode() : egr::GradNodeBase() {}
  AngleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AngleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AngleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AngleGradNode>(new AngleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class ArgsortGradNode : public egr::GradNodeBase {
 public:
  ArgsortGradNode() : egr::GradNodeBase() {}
  ArgsortGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ArgsortGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ArgsortGradNode"; }

  void ClearTensorWrappers() override {
    indices_.clear();
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ArgsortGradNode>(new ArgsortGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperindices(const paddle::experimental::Tensor& indices) {
    indices_ = egr::TensorWrapper(indices, false);
  }
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }
  void SetAttributedescending(const bool& descending) {
    descending_ = descending;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper indices_;
  egr::TensorWrapper x_;

  // Attributes
  int axis_;
  bool descending_;
};

class AsComplexGradNode : public egr::GradNodeBase {
 public:
  AsComplexGradNode() : egr::GradNodeBase() {}
  AsComplexGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AsComplexGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AsComplexGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<AsComplexGradNode>(new AsComplexGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes

 private:
  // TensorWrappers

  // Attributes
};

class AsRealGradNode : public egr::GradNodeBase {
 public:
  AsRealGradNode() : egr::GradNodeBase() {}
  AsRealGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AsRealGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AsRealGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<AsRealGradNode>(new AsRealGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes

 private:
  // TensorWrappers

  // Attributes
};

class AsinGradNode : public egr::GradNodeBase {
 public:
  AsinGradNode() : egr::GradNodeBase() {}
  AsinGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AsinGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AsinGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AsinGradNode>(new AsinGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class AsinhGradNode : public egr::GradNodeBase {
 public:
  AsinhGradNode() : egr::GradNodeBase() {}
  AsinhGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AsinhGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AsinhGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AsinhGradNode>(new AsinhGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class AtanGradNode : public egr::GradNodeBase {
 public:
  AtanGradNode() : egr::GradNodeBase() {}
  AtanGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AtanGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AtanGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AtanGradNode>(new AtanGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class Atan2GradNode : public egr::GradNodeBase {
 public:
  Atan2GradNode() : egr::GradNodeBase() {}
  Atan2GradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Atan2GradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Atan2GradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<Atan2GradNode>(new Atan2GradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
};

class AtanhGradNode : public egr::GradNodeBase {
 public:
  AtanhGradNode() : egr::GradNodeBase() {}
  AtanhGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AtanhGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AtanhGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AtanhGradNode>(new AtanhGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class BmmGradNode : public egr::GradNodeBase {
 public:
  BmmGradNode() : egr::GradNodeBase() {}
  BmmGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~BmmGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "BmmGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<BmmGradNode>(new BmmGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
};

class CeilGradNode : public egr::GradNodeBase {
 public:
  CeilGradNode() : egr::GradNodeBase() {}
  CeilGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~CeilGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "CeilGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<CeilGradNode>(new CeilGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes

 private:
  // TensorWrappers

  // Attributes
};

class CeluGradNode : public egr::GradNodeBase {
 public:
  CeluGradNode() : egr::GradNodeBase() {}
  CeluGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~CeluGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "CeluGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<CeluGradNode>(new CeluGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributealpha(const float& alpha) { alpha_ = alpha; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  float alpha_;
};

class CeluDoubleGradNode : public egr::GradNodeBase {
 public:
  CeluDoubleGradNode() : egr::GradNodeBase() {}
  CeluDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~CeluDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "CeluDoubleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<CeluDoubleGradNode>(new CeluDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappergrad_out(const paddle::experimental::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }

  // SetAttributes
  void SetAttributealpha(const float& alpha) { alpha_ = alpha; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper grad_out_;

  // Attributes
  float alpha_;
};

class CholeskyGradNode : public egr::GradNodeBase {
 public:
  CholeskyGradNode() : egr::GradNodeBase() {}
  CholeskyGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~CholeskyGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "CholeskyGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<CholeskyGradNode>(new CholeskyGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributeupper(const bool& upper) { upper_ = upper; }

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
  bool upper_;
};

class CholeskySolveGradNode : public egr::GradNodeBase {
 public:
  CholeskySolveGradNode() : egr::GradNodeBase() {}
  CholeskySolveGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~CholeskySolveGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "CholeskySolveGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<CholeskySolveGradNode>(
        new CholeskySolveGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributeupper(const bool& upper) { upper_ = upper; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;
  egr::TensorWrapper out_;

  // Attributes
  bool upper_;
};

class ClipGradNode : public egr::GradNodeBase {
 public:
  ClipGradNode() : egr::GradNodeBase() {}
  ClipGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ClipGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ClipGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<ClipGradNode>(new ClipGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributemin(const paddle::experimental::Scalar& min) { min_ = min; }
  void SetAttributemax(const paddle::experimental::Scalar& max) { max_ = max; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  paddle::experimental::Scalar min_ = 0.;
  paddle::experimental::Scalar max_ = 0.;
};

class ClipDoubleGradNode : public egr::GradNodeBase {
 public:
  ClipDoubleGradNode() : egr::GradNodeBase() {}
  ClipDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ClipDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ClipDoubleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ClipDoubleGradNode>(new ClipDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributemin(const paddle::experimental::Scalar& min) { min_ = min; }
  void SetAttributemax(const paddle::experimental::Scalar& max) { max_ = max; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  paddle::experimental::Scalar min_ = 0.;
  paddle::experimental::Scalar max_ = 0.;
};

class ComplexGradNode : public egr::GradNodeBase {
 public:
  ComplexGradNode() : egr::GradNodeBase() {}
  ComplexGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ComplexGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ComplexGradNode"; }

  void ClearTensorWrappers() override {
    real_.clear();
    imag_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ComplexGradNode>(new ComplexGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperreal(const paddle::experimental::Tensor& real) {
    real_ = egr::TensorWrapper(real, false);
  }
  void SetTensorWrapperimag(const paddle::experimental::Tensor& imag) {
    imag_ = egr::TensorWrapper(imag, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper real_;
  egr::TensorWrapper imag_;

  // Attributes
};

class ConjGradNode : public egr::GradNodeBase {
 public:
  ConjGradNode() : egr::GradNodeBase() {}
  ConjGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ConjGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ConjGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<ConjGradNode>(new ConjGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes

 private:
  // TensorWrappers

  // Attributes
};

class CosGradNode : public egr::GradNodeBase {
 public:
  CosGradNode() : egr::GradNodeBase() {}
  CosGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~CosGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "CosGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<CosGradNode>(new CosGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class CosDoubleGradNode : public egr::GradNodeBase {
 public:
  CosDoubleGradNode() : egr::GradNodeBase() {}
  CosDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~CosDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "CosDoubleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<CosDoubleGradNode>(new CosDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappergrad_out(const paddle::experimental::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper grad_out_;

  // Attributes
};

class CosTripleGradNode : public egr::GradNodeBase {
 public:
  CosTripleGradNode() : egr::GradNodeBase() {}
  CosTripleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~CosTripleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "CosTripleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    grad_out_forward_.clear();
    grad_x_grad_forward_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<CosTripleGradNode>(new CosTripleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappergrad_out_forward(
      const paddle::experimental::Tensor& grad_out_forward) {
    grad_out_forward_ = egr::TensorWrapper(grad_out_forward, false);
  }
  void SetTensorWrappergrad_x_grad_forward(
      const paddle::experimental::Tensor& grad_x_grad_forward) {
    grad_x_grad_forward_ = egr::TensorWrapper(grad_x_grad_forward, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper grad_out_forward_;
  egr::TensorWrapper grad_x_grad_forward_;

  // Attributes
};

class CoshGradNode : public egr::GradNodeBase {
 public:
  CoshGradNode() : egr::GradNodeBase() {}
  CoshGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~CoshGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "CoshGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<CoshGradNode>(new CoshGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class CropGradNode : public egr::GradNodeBase {
 public:
  CropGradNode() : egr::GradNodeBase() {}
  CropGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~CropGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "CropGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<CropGradNode>(new CropGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributeoffsets(const paddle::experimental::IntArray& offsets) {
    offsets_ = offsets;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  paddle::experimental::IntArray offsets_;
};

class CrossGradNode : public egr::GradNodeBase {
 public:
  CrossGradNode() : egr::GradNodeBase() {}
  CrossGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~CrossGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "CrossGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<CrossGradNode>(new CrossGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
  int axis_;
};

class DetGradNode : public egr::GradNodeBase {
 public:
  DetGradNode() : egr::GradNodeBase() {}
  DetGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~DetGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "DetGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<DetGradNode>(new DetGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_;

  // Attributes
};

class DiagGradNode : public egr::GradNodeBase {
 public:
  DiagGradNode() : egr::GradNodeBase() {}
  DiagGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~DiagGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "DiagGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<DiagGradNode>(new DiagGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }

  // SetAttributes
  void SetAttributeoffset(const int& offset) { offset_ = offset; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  int offset_;
};

class DiagonalGradNode : public egr::GradNodeBase {
 public:
  DiagonalGradNode() : egr::GradNodeBase() {}
  DiagonalGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~DiagonalGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "DiagonalGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<DiagonalGradNode>(new DiagonalGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }

  // SetAttributes
  void SetAttributeoffset(const int& offset) { offset_ = offset; }
  void SetAttributeaxis1(const int& axis1) { axis1_ = axis1; }
  void SetAttributeaxis2(const int& axis2) { axis2_ = axis2; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  int offset_ = 0;
  int axis1_ = 0;
  int axis2_ = 1;
};

class DigammaGradNode : public egr::GradNodeBase {
 public:
  DigammaGradNode() : egr::GradNodeBase() {}
  DigammaGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~DigammaGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "DigammaGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<DigammaGradNode>(new DigammaGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class DistGradNode : public egr::GradNodeBase {
 public:
  DistGradNode() : egr::GradNodeBase() {}
  DistGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~DistGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "DistGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<DistGradNode>(new DistGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributep(const float& p) { p_ = p; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;
  egr::TensorWrapper out_;

  // Attributes
  float p_;
};

class DotGradNode : public egr::GradNodeBase {
 public:
  DotGradNode() : egr::GradNodeBase() {}
  DotGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~DotGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "DotGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<DotGradNode>(new DotGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
};

class EigGradNode : public egr::GradNodeBase {
 public:
  EigGradNode() : egr::GradNodeBase() {}
  EigGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~EigGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "EigGradNode"; }

  void ClearTensorWrappers() override {
    out_w_.clear();
    out_v_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<EigGradNode>(new EigGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout_w(const paddle::experimental::Tensor& out_w) {
    out_w_ = egr::TensorWrapper(out_w, false);
  }
  void SetTensorWrapperout_v(const paddle::experimental::Tensor& out_v) {
    out_v_ = egr::TensorWrapper(out_v, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_w_;
  egr::TensorWrapper out_v_;

  // Attributes
};

class EighGradNode : public egr::GradNodeBase {
 public:
  EighGradNode() : egr::GradNodeBase() {}
  EighGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~EighGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "EighGradNode"; }

  void ClearTensorWrappers() override {
    out_w_.clear();
    out_v_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<EighGradNode>(new EighGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout_w(const paddle::experimental::Tensor& out_w) {
    out_w_ = egr::TensorWrapper(out_w, false);
  }
  void SetTensorWrapperout_v(const paddle::experimental::Tensor& out_v) {
    out_v_ = egr::TensorWrapper(out_v, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_w_;
  egr::TensorWrapper out_v_;

  // Attributes
};

class EluGradNode : public egr::GradNodeBase {
 public:
  EluGradNode() : egr::GradNodeBase() {}
  EluGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~EluGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "EluGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<EluGradNode>(new EluGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributealpha(const float& alpha) { alpha_ = alpha; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_;

  // Attributes
  float alpha_;
};

class EluDoubleGradNode : public egr::GradNodeBase {
 public:
  EluDoubleGradNode() : egr::GradNodeBase() {}
  EluDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~EluDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "EluDoubleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<EluDoubleGradNode>(new EluDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappergrad_out(const paddle::experimental::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }

  // SetAttributes
  void SetAttributealpha(const float& alpha) { alpha_ = alpha; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper grad_out_;

  // Attributes
  float alpha_;
};

class ErfGradNode : public egr::GradNodeBase {
 public:
  ErfGradNode() : egr::GradNodeBase() {}
  ErfGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ErfGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ErfGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<ErfGradNode>(new ErfGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class ErfinvGradNode : public egr::GradNodeBase {
 public:
  ErfinvGradNode() : egr::GradNodeBase() {}
  ErfinvGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ErfinvGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ErfinvGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ErfinvGradNode>(new ErfinvGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
};

class ExpGradNode : public egr::GradNodeBase {
 public:
  ExpGradNode() : egr::GradNodeBase() {}
  ExpGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ExpGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ExpGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<ExpGradNode>(new ExpGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
};

class Expm1GradNode : public egr::GradNodeBase {
 public:
  Expm1GradNode() : egr::GradNodeBase() {}
  Expm1GradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Expm1GradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Expm1GradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<Expm1GradNode>(new Expm1GradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
};

class FftC2cGradNode : public egr::GradNodeBase {
 public:
  FftC2cGradNode() : egr::GradNodeBase() {}
  FftC2cGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~FftC2cGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "FftC2cGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<FftC2cGradNode>(new FftC2cGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeaxes(const std::vector<int64_t>& axes) { axes_ = axes; }
  void SetAttributenormalization(const std::string& normalization) {
    normalization_ = normalization;
  }
  void SetAttributeforward(const bool& forward) { forward_ = forward; }

 private:
  // TensorWrappers

  // Attributes
  std::vector<int64_t> axes_;
  std::string normalization_;
  bool forward_;
};

class FftC2rGradNode : public egr::GradNodeBase {
 public:
  FftC2rGradNode() : egr::GradNodeBase() {}
  FftC2rGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~FftC2rGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "FftC2rGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<FftC2rGradNode>(new FftC2rGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeaxes(const std::vector<int64_t>& axes) { axes_ = axes; }
  void SetAttributenormalization(const std::string& normalization) {
    normalization_ = normalization;
  }
  void SetAttributeforward(const bool& forward) { forward_ = forward; }
  void SetAttributelast_dim_size(const int64_t& last_dim_size) {
    last_dim_size_ = last_dim_size;
  }

 private:
  // TensorWrappers

  // Attributes
  std::vector<int64_t> axes_;
  std::string normalization_;
  bool forward_;
  int64_t last_dim_size_;
};

class FftR2cGradNode : public egr::GradNodeBase {
 public:
  FftR2cGradNode() : egr::GradNodeBase() {}
  FftR2cGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~FftR2cGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "FftR2cGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<FftR2cGradNode>(new FftR2cGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }

  // SetAttributes
  void SetAttributeaxes(const std::vector<int64_t>& axes) { axes_ = axes; }
  void SetAttributenormalization(const std::string& normalization) {
    normalization_ = normalization;
  }
  void SetAttributeforward(const bool& forward) { forward_ = forward; }
  void SetAttributeonesided(const bool& onesided) { onesided_ = onesided; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  std::vector<int64_t> axes_;
  std::string normalization_;
  bool forward_;
  bool onesided_;
};

class FillDiagonalGradNode : public egr::GradNodeBase {
 public:
  FillDiagonalGradNode() : egr::GradNodeBase() {}
  FillDiagonalGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~FillDiagonalGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "FillDiagonalGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<FillDiagonalGradNode>(new FillDiagonalGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributevalue(const float& value) { value_ = value; }
  void SetAttributeoffset(const int& offset) { offset_ = offset; }
  void SetAttributewrap(const bool& wrap) { wrap_ = wrap; }

 private:
  // TensorWrappers

  // Attributes
  float value_;
  int offset_;
  bool wrap_;
};

class FillDiagonalTensorGradNode : public egr::GradNodeBase {
 public:
  FillDiagonalTensorGradNode() : egr::GradNodeBase() {}
  FillDiagonalTensorGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~FillDiagonalTensorGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "FillDiagonalTensorGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<FillDiagonalTensorGradNode>(
        new FillDiagonalTensorGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeoffset(const int64_t& offset) { offset_ = offset; }
  void SetAttributedim1(const int& dim1) { dim1_ = dim1; }
  void SetAttributedim2(const int& dim2) { dim2_ = dim2; }

 private:
  // TensorWrappers

  // Attributes
  int64_t offset_;
  int dim1_;
  int dim2_;
};

class FlipGradNode : public egr::GradNodeBase {
 public:
  FlipGradNode() : egr::GradNodeBase() {}
  FlipGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~FlipGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "FlipGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<FlipGradNode>(new FlipGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeaxis(const std::vector<int>& axis) { axis_ = axis; }

 private:
  // TensorWrappers

  // Attributes
  std::vector<int> axis_;
};

class FloorGradNode : public egr::GradNodeBase {
 public:
  FloorGradNode() : egr::GradNodeBase() {}
  FloorGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~FloorGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "FloorGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<FloorGradNode>(new FloorGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes

 private:
  // TensorWrappers

  // Attributes
};

class FoldGradNode : public egr::GradNodeBase {
 public:
  FoldGradNode() : egr::GradNodeBase() {}
  FoldGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~FoldGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "FoldGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<FoldGradNode>(new FoldGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }

  // SetAttributes
  void SetAttributeoutput_sizes(const std::vector<int>& output_sizes) {
    output_sizes_ = output_sizes;
  }
  void SetAttributekernel_sizes(const std::vector<int>& kernel_sizes) {
    kernel_sizes_ = kernel_sizes;
  }
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributedilations(const std::vector<int>& dilations) {
    dilations_ = dilations;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  std::vector<int> output_sizes_;
  std::vector<int> kernel_sizes_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<int> dilations_;
};

class FrameGradNode : public egr::GradNodeBase {
 public:
  FrameGradNode() : egr::GradNodeBase() {}
  FrameGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~FrameGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "FrameGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<FrameGradNode>(new FrameGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributeframe_length(const int& frame_length) {
    frame_length_ = frame_length;
  }
  void SetAttributehop_length(const int& hop_length) {
    hop_length_ = hop_length;
  }
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  int frame_length_;
  int hop_length_;
  int axis_;
};

class GatherNdGradNode : public egr::GradNodeBase {
 public:
  GatherNdGradNode() : egr::GradNodeBase() {}
  GatherNdGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~GatherNdGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "GatherNdGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    index_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<GatherNdGradNode>(new GatherNdGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }
  void SetTensorWrapperindex(const paddle::experimental::Tensor& index) {
    index_ = egr::TensorWrapper(index, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper index_;

  // Attributes
};

class GeluGradNode : public egr::GradNodeBase {
 public:
  GeluGradNode() : egr::GradNodeBase() {}
  GeluGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~GeluGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "GeluGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<GeluGradNode>(new GeluGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributeapproximate(const bool& approximate) {
    approximate_ = approximate;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  bool approximate_;
};

class GridSampleGradNode : public egr::GradNodeBase {
 public:
  GridSampleGradNode() : egr::GradNodeBase() {}
  GridSampleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~GridSampleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "GridSampleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    grid_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<GridSampleGradNode>(new GridSampleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappergrid(const paddle::experimental::Tensor& grid) {
    grid_ = egr::TensorWrapper(grid, false);
  }

  // SetAttributes
  void SetAttributemode(const std::string& mode) { mode_ = mode; }
  void SetAttributepadding_mode(const std::string& padding_mode) {
    padding_mode_ = padding_mode;
  }
  void SetAttributealign_corners(const bool& align_corners) {
    align_corners_ = align_corners;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper grid_;

  // Attributes
  std::string mode_;
  std::string padding_mode_;
  bool align_corners_;
};

class GumbelSoftmaxGradNode : public egr::GradNodeBase {
 public:
  GumbelSoftmaxGradNode() : egr::GradNodeBase() {}
  GumbelSoftmaxGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~GumbelSoftmaxGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "GumbelSoftmaxGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<GumbelSoftmaxGradNode>(
        new GumbelSoftmaxGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
  int axis_;
};

class HardshrinkGradNode : public egr::GradNodeBase {
 public:
  HardshrinkGradNode() : egr::GradNodeBase() {}
  HardshrinkGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~HardshrinkGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "HardshrinkGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<HardshrinkGradNode>(new HardshrinkGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributethreshold(const float& threshold) { threshold_ = threshold; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  float threshold_;
};

class HardsigmoidGradNode : public egr::GradNodeBase {
 public:
  HardsigmoidGradNode() : egr::GradNodeBase() {}
  HardsigmoidGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~HardsigmoidGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "HardsigmoidGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<HardsigmoidGradNode>(new HardsigmoidGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributeslope(const float& slope) { slope_ = slope; }
  void SetAttributeoffset(const float& offset) { offset_ = offset; }

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
  float slope_;
  float offset_;
};

class HardtanhGradNode : public egr::GradNodeBase {
 public:
  HardtanhGradNode() : egr::GradNodeBase() {}
  HardtanhGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~HardtanhGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "HardtanhGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<HardtanhGradNode>(new HardtanhGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributet_min(const float& t_min) { t_min_ = t_min; }
  void SetAttributet_max(const float& t_max) { t_max_ = t_max; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  float t_min_;
  float t_max_;
};

class ImagGradNode : public egr::GradNodeBase {
 public:
  ImagGradNode() : egr::GradNodeBase() {}
  ImagGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ImagGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ImagGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<ImagGradNode>(new ImagGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes

 private:
  // TensorWrappers

  // Attributes
};

class IndexSampleGradNode : public egr::GradNodeBase {
 public:
  IndexSampleGradNode() : egr::GradNodeBase() {}
  IndexSampleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~IndexSampleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "IndexSampleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    index_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<IndexSampleGradNode>(new IndexSampleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }
  void SetTensorWrapperindex(const paddle::experimental::Tensor& index) {
    index_ = egr::TensorWrapper(index, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper index_;

  // Attributes
};

class IndexSelectGradNode : public egr::GradNodeBase {
 public:
  IndexSelectGradNode() : egr::GradNodeBase() {}
  IndexSelectGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~IndexSelectGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "IndexSelectGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    index_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<IndexSelectGradNode>(new IndexSelectGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }
  void SetTensorWrapperindex(const paddle::experimental::Tensor& index) {
    index_ = egr::TensorWrapper(index, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper index_;

  // Attributes
  int axis_;
};

class InverseGradNode : public egr::GradNodeBase {
 public:
  InverseGradNode() : egr::GradNodeBase() {}
  InverseGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~InverseGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "InverseGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<InverseGradNode>(new InverseGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
};

class KthvalueGradNode : public egr::GradNodeBase {
 public:
  KthvalueGradNode() : egr::GradNodeBase() {}
  KthvalueGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~KthvalueGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "KthvalueGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    indices_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<KthvalueGradNode>(new KthvalueGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperindices(const paddle::experimental::Tensor& indices) {
    indices_ = egr::TensorWrapper(indices, false);
  }

  // SetAttributes
  void SetAttributek(const int& k) { k_ = k; }
  void SetAttributeaxis(const int& axis) { axis_ = axis; }
  void SetAttributekeepdim(const bool& keepdim) { keepdim_ = keepdim; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper indices_;

  // Attributes
  int k_;
  int axis_;
  bool keepdim_;
};

class LabelSmoothGradNode : public egr::GradNodeBase {
 public:
  LabelSmoothGradNode() : egr::GradNodeBase() {}
  LabelSmoothGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~LabelSmoothGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "LabelSmoothGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<LabelSmoothGradNode>(new LabelSmoothGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeepsilon(const float& epsilon) { epsilon_ = epsilon; }

 private:
  // TensorWrappers

  // Attributes
  float epsilon_;
};

class LeakyReluGradNode : public egr::GradNodeBase {
 public:
  LeakyReluGradNode() : egr::GradNodeBase() {}
  LeakyReluGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~LeakyReluGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "LeakyReluGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<LeakyReluGradNode>(new LeakyReluGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributenegative_slope(const float& negative_slope) {
    negative_slope_ = negative_slope;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  float negative_slope_;
};

class LeakyReluDoubleGradNode : public egr::GradNodeBase {
 public:
  LeakyReluDoubleGradNode() : egr::GradNodeBase() {}
  LeakyReluDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~LeakyReluDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "LeakyReluDoubleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<LeakyReluDoubleGradNode>(
        new LeakyReluDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributenegative_slope(const float& negative_slope) {
    negative_slope_ = negative_slope;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  float negative_slope_;
};

class LerpGradNode : public egr::GradNodeBase {
 public:
  LerpGradNode() : egr::GradNodeBase() {}
  LerpGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~LerpGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "LerpGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();
    weight_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<LerpGradNode>(new LerpGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }
  void SetTensorWrapperweight(const paddle::experimental::Tensor& weight) {
    weight_ = egr::TensorWrapper(weight, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;
  egr::TensorWrapper weight_;
  egr::TensorWrapper out_;

  // Attributes
};

class LgammaGradNode : public egr::GradNodeBase {
 public:
  LgammaGradNode() : egr::GradNodeBase() {}
  LgammaGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~LgammaGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "LgammaGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<LgammaGradNode>(new LgammaGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class LogGradNode : public egr::GradNodeBase {
 public:
  LogGradNode() : egr::GradNodeBase() {}
  LogGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~LogGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "LogGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<LogGradNode>(new LogGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class LogDoubleGradNode : public egr::GradNodeBase {
 public:
  LogDoubleGradNode() : egr::GradNodeBase() {}
  LogDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~LogDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "LogDoubleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<LogDoubleGradNode>(new LogDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappergrad_out(const paddle::experimental::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper grad_out_;

  // Attributes
};

class Log10GradNode : public egr::GradNodeBase {
 public:
  Log10GradNode() : egr::GradNodeBase() {}
  Log10GradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Log10GradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Log10GradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<Log10GradNode>(new Log10GradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class Log1pGradNode : public egr::GradNodeBase {
 public:
  Log1pGradNode() : egr::GradNodeBase() {}
  Log1pGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Log1pGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Log1pGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<Log1pGradNode>(new Log1pGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class Log2GradNode : public egr::GradNodeBase {
 public:
  Log2GradNode() : egr::GradNodeBase() {}
  Log2GradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Log2GradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Log2GradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<Log2GradNode>(new Log2GradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class LogLossGradNode : public egr::GradNodeBase {
 public:
  LogLossGradNode() : egr::GradNodeBase() {}
  LogLossGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~LogLossGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "LogLossGradNode"; }

  void ClearTensorWrappers() override {
    input_.clear();
    label_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<LogLossGradNode>(new LogLossGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperinput(const paddle::experimental::Tensor& input) {
    input_ = egr::TensorWrapper(input, false);
  }
  void SetTensorWrapperlabel(const paddle::experimental::Tensor& label) {
    label_ = egr::TensorWrapper(label, false);
  }

  // SetAttributes
  void SetAttributeepsilon(const float& epsilon) { epsilon_ = epsilon; }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;
  egr::TensorWrapper label_;

  // Attributes
  float epsilon_;
};

class LogitGradNode : public egr::GradNodeBase {
 public:
  LogitGradNode() : egr::GradNodeBase() {}
  LogitGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~LogitGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "LogitGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<LogitGradNode>(new LogitGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributeeps(const float& eps) { eps_ = eps; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  float eps_;
};

class LogsigmoidGradNode : public egr::GradNodeBase {
 public:
  LogsigmoidGradNode() : egr::GradNodeBase() {}
  LogsigmoidGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~LogsigmoidGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "LogsigmoidGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<LogsigmoidGradNode>(new LogsigmoidGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class LuUnpackGradNode : public egr::GradNodeBase {
 public:
  LuUnpackGradNode() : egr::GradNodeBase() {}
  LuUnpackGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~LuUnpackGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "LuUnpackGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();
    l_.clear();
    u_.clear();
    pmat_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<LuUnpackGradNode>(new LuUnpackGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }
  void SetTensorWrapperl(const paddle::experimental::Tensor& l) {
    l_ = egr::TensorWrapper(l, false);
  }
  void SetTensorWrapperu(const paddle::experimental::Tensor& u) {
    u_ = egr::TensorWrapper(u, false);
  }
  void SetTensorWrapperpmat(const paddle::experimental::Tensor& pmat) {
    pmat_ = egr::TensorWrapper(pmat, false);
  }

  // SetAttributes
  void SetAttributeunpack_ludata(const bool& unpack_ludata) {
    unpack_ludata_ = unpack_ludata;
  }
  void SetAttributeunpack_pivots(const bool& unpack_pivots) {
    unpack_pivots_ = unpack_pivots;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;
  egr::TensorWrapper l_;
  egr::TensorWrapper u_;
  egr::TensorWrapper pmat_;

  // Attributes
  bool unpack_ludata_;
  bool unpack_pivots_;
};

class MaskedSelectGradNode : public egr::GradNodeBase {
 public:
  MaskedSelectGradNode() : egr::GradNodeBase() {}
  MaskedSelectGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MaskedSelectGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MaskedSelectGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    mask_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MaskedSelectGradNode>(new MaskedSelectGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }
  void SetTensorWrappermask(const paddle::experimental::Tensor& mask) {
    mask_ = egr::TensorWrapper(mask, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper mask_;

  // Attributes
};

class MatrixPowerGradNode : public egr::GradNodeBase {
 public:
  MatrixPowerGradNode() : egr::GradNodeBase() {}
  MatrixPowerGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MatrixPowerGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MatrixPowerGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MatrixPowerGradNode>(new MatrixPowerGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributen(const int& n) { n_ = n; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_;

  // Attributes
  int n_;
};

class MaxoutGradNode : public egr::GradNodeBase {
 public:
  MaxoutGradNode() : egr::GradNodeBase() {}
  MaxoutGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MaxoutGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MaxoutGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MaxoutGradNode>(new MaxoutGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributegroups(const int& groups) { groups_ = groups; }
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_;

  // Attributes
  int groups_;
  int axis_;
};

class ModeGradNode : public egr::GradNodeBase {
 public:
  ModeGradNode() : egr::GradNodeBase() {}
  ModeGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ModeGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ModeGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    indices_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<ModeGradNode>(new ModeGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperindices(const paddle::experimental::Tensor& indices) {
    indices_ = egr::TensorWrapper(indices, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }
  void SetAttributekeepdim(const bool& keepdim) { keepdim_ = keepdim; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper indices_;

  // Attributes
  int axis_;
  bool keepdim_;
};

class MvGradNode : public egr::GradNodeBase {
 public:
  MvGradNode() : egr::GradNodeBase() {}
  MvGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MvGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MvGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    vec_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<MvGradNode>(new MvGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappervec(const paddle::experimental::Tensor& vec) {
    vec_ = egr::TensorWrapper(vec, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper vec_;

  // Attributes
};

class NllLossGradNode : public egr::GradNodeBase {
 public:
  NllLossGradNode() : egr::GradNodeBase() {}
  NllLossGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~NllLossGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "NllLossGradNode"; }

  void ClearTensorWrappers() override {
    input_.clear();
    label_.clear();
    weight_.clear();
    total_weight_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<NllLossGradNode>(new NllLossGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperinput(const paddle::experimental::Tensor& input) {
    input_ = egr::TensorWrapper(input, false);
  }
  void SetTensorWrapperlabel(const paddle::experimental::Tensor& label) {
    label_ = egr::TensorWrapper(label, false);
  }
  void SetTensorWrapperweight(const paddle::experimental::Tensor& weight) {
    weight_ = egr::TensorWrapper(weight, false);
  }
  void SetTensorWrappertotal_weight(
      const paddle::experimental::Tensor& total_weight) {
    total_weight_ = egr::TensorWrapper(total_weight, false);
  }

  // SetAttributes
  void SetAttributeignore_index(const int64_t& ignore_index) {
    ignore_index_ = ignore_index;
  }
  void SetAttributereduction(const std::string& reduction) {
    reduction_ = reduction;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;
  egr::TensorWrapper label_;
  egr::TensorWrapper weight_;
  egr::TensorWrapper total_weight_;

  // Attributes
  int64_t ignore_index_;
  std::string reduction_;
};

class OverlapAddGradNode : public egr::GradNodeBase {
 public:
  OverlapAddGradNode() : egr::GradNodeBase() {}
  OverlapAddGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~OverlapAddGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "OverlapAddGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<OverlapAddGradNode>(new OverlapAddGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributehop_length(const int& hop_length) {
    hop_length_ = hop_length;
  }
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  int hop_length_;
  int axis_;
};

class PixelShuffleGradNode : public egr::GradNodeBase {
 public:
  PixelShuffleGradNode() : egr::GradNodeBase() {}
  PixelShuffleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~PixelShuffleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "PixelShuffleGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<PixelShuffleGradNode>(new PixelShuffleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeupscale_factor(const int& upscale_factor) {
    upscale_factor_ = upscale_factor;
  }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }

 private:
  // TensorWrappers

  // Attributes
  int upscale_factor_;
  std::string data_format_;
};

class PoissonGradNode : public egr::GradNodeBase {
 public:
  PoissonGradNode() : egr::GradNodeBase() {}
  PoissonGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~PoissonGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "PoissonGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<PoissonGradNode>(new PoissonGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes

 private:
  // TensorWrappers

  // Attributes
};

class PutAlongAxisGradNode : public egr::GradNodeBase {
 public:
  PutAlongAxisGradNode() : egr::GradNodeBase() {}
  PutAlongAxisGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~PutAlongAxisGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "PutAlongAxisGradNode"; }

  void ClearTensorWrappers() override {
    arr_.clear();
    indices_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<PutAlongAxisGradNode>(new PutAlongAxisGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperarr(const paddle::experimental::Tensor& arr) {
    arr_ = egr::TensorWrapper(arr, false);
  }
  void SetTensorWrapperindices(const paddle::experimental::Tensor& indices) {
    indices_ = egr::TensorWrapper(indices, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }
  void SetAttributereduce(const std::string& reduce) { reduce_ = reduce; }

 private:
  // TensorWrappers
  egr::TensorWrapper arr_;
  egr::TensorWrapper indices_;

  // Attributes
  int axis_;
  std::string reduce_;
};

class QrGradNode : public egr::GradNodeBase {
 public:
  QrGradNode() : egr::GradNodeBase() {}
  QrGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~QrGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "QrGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    q_.clear();
    r_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<QrGradNode>(new QrGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperq(const paddle::experimental::Tensor& q) {
    q_ = egr::TensorWrapper(q, false);
  }
  void SetTensorWrapperr(const paddle::experimental::Tensor& r) {
    r_ = egr::TensorWrapper(r, false);
  }

  // SetAttributes
  void SetAttributemode(const std::string& mode) { mode_ = mode; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper q_;
  egr::TensorWrapper r_;

  // Attributes
  std::string mode_;
};

class RealGradNode : public egr::GradNodeBase {
 public:
  RealGradNode() : egr::GradNodeBase() {}
  RealGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~RealGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "RealGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<RealGradNode>(new RealGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes

 private:
  // TensorWrappers

  // Attributes
};

class ReciprocalGradNode : public egr::GradNodeBase {
 public:
  ReciprocalGradNode() : egr::GradNodeBase() {}
  ReciprocalGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ReciprocalGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ReciprocalGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ReciprocalGradNode>(new ReciprocalGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
};

class ReluGradNode : public egr::GradNodeBase {
 public:
  ReluGradNode() : egr::GradNodeBase() {}
  ReluGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ReluGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ReluGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<ReluGradNode>(new ReluGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
};

class ReluDoubleGradNode : public egr::GradNodeBase {
 public:
  ReluDoubleGradNode() : egr::GradNodeBase() {}
  ReluDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ReluDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ReluDoubleGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ReluDoubleGradNode>(new ReluDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
};

class RenormGradNode : public egr::GradNodeBase {
 public:
  RenormGradNode() : egr::GradNodeBase() {}
  RenormGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~RenormGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "RenormGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<RenormGradNode>(new RenormGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributep(const float& p) { p_ = p; }
  void SetAttributeaxis(const int& axis) { axis_ = axis; }
  void SetAttributemax_norm(const float& max_norm) { max_norm_ = max_norm; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  float p_;
  int axis_;
  float max_norm_;
};

class RollGradNode : public egr::GradNodeBase {
 public:
  RollGradNode() : egr::GradNodeBase() {}
  RollGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~RollGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "RollGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<RollGradNode>(new RollGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }

  // SetAttributes
  void SetAttributeshifts(const paddle::experimental::IntArray& shifts) {
    shifts_ = shifts;
  }
  void SetAttributeaxis(const std::vector<int64_t>& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  paddle::experimental::IntArray shifts_;
  std::vector<int64_t> axis_;
};

class RoundGradNode : public egr::GradNodeBase {
 public:
  RoundGradNode() : egr::GradNodeBase() {}
  RoundGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~RoundGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "RoundGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<RoundGradNode>(new RoundGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes

 private:
  // TensorWrappers

  // Attributes
};

class RsqrtGradNode : public egr::GradNodeBase {
 public:
  RsqrtGradNode() : egr::GradNodeBase() {}
  RsqrtGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~RsqrtGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "RsqrtGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<RsqrtGradNode>(new RsqrtGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
};

class RsqrtDoubleGradNode : public egr::GradNodeBase {
 public:
  RsqrtDoubleGradNode() : egr::GradNodeBase() {}
  RsqrtDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~RsqrtDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "RsqrtDoubleGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();
    grad_x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<RsqrtDoubleGradNode>(new RsqrtDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }
  void SetTensorWrappergrad_x(const paddle::experimental::Tensor& grad_x) {
    grad_x_ = egr::TensorWrapper(grad_x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;
  egr::TensorWrapper grad_x_;

  // Attributes
};

class ScatterGradNode : public egr::GradNodeBase {
 public:
  ScatterGradNode() : egr::GradNodeBase() {}
  ScatterGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ScatterGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ScatterGradNode"; }

  void ClearTensorWrappers() override {
    index_.clear();
    updates_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ScatterGradNode>(new ScatterGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperindex(const paddle::experimental::Tensor& index) {
    index_ = egr::TensorWrapper(index, false);
  }
  void SetTensorWrapperupdates(const paddle::experimental::Tensor& updates) {
    updates_ = egr::TensorWrapper(updates, true);
  }

  // SetAttributes
  void SetAttributeoverwrite(const bool& overwrite) { overwrite_ = overwrite; }

 private:
  // TensorWrappers
  egr::TensorWrapper index_;
  egr::TensorWrapper updates_;

  // Attributes
  bool overwrite_;
};

class ScatterNdAddGradNode : public egr::GradNodeBase {
 public:
  ScatterNdAddGradNode() : egr::GradNodeBase() {}
  ScatterNdAddGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ScatterNdAddGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ScatterNdAddGradNode"; }

  void ClearTensorWrappers() override {
    index_.clear();
    updates_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ScatterNdAddGradNode>(new ScatterNdAddGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperindex(const paddle::experimental::Tensor& index) {
    index_ = egr::TensorWrapper(index, false);
  }
  void SetTensorWrapperupdates(const paddle::experimental::Tensor& updates) {
    updates_ = egr::TensorWrapper(updates, true);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper index_;
  egr::TensorWrapper updates_;

  // Attributes
};

class SeluGradNode : public egr::GradNodeBase {
 public:
  SeluGradNode() : egr::GradNodeBase() {}
  SeluGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SeluGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SeluGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SeluGradNode>(new SeluGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributescale(const float& scale) { scale_ = scale; }
  void SetAttributealpha(const float& alpha) { alpha_ = alpha; }

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
  float scale_;
  float alpha_;
};

class SendUvGradNode : public egr::GradNodeBase {
 public:
  SendUvGradNode() : egr::GradNodeBase() {}
  SendUvGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SendUvGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SendUvGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();
    src_index_.clear();
    dst_index_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SendUvGradNode>(new SendUvGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }
  void SetTensorWrappersrc_index(
      const paddle::experimental::Tensor& src_index) {
    src_index_ = egr::TensorWrapper(src_index, false);
  }
  void SetTensorWrapperdst_index(
      const paddle::experimental::Tensor& dst_index) {
    dst_index_ = egr::TensorWrapper(dst_index, false);
  }

  // SetAttributes
  void SetAttributemessage_op(const std::string& message_op) {
    message_op_ = message_op;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;
  egr::TensorWrapper src_index_;
  egr::TensorWrapper dst_index_;

  // Attributes
  std::string message_op_ = "ADD";
};

class SigmoidGradNode : public egr::GradNodeBase {
 public:
  SigmoidGradNode() : egr::GradNodeBase() {}
  SigmoidGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SigmoidGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SigmoidGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SigmoidGradNode>(new SigmoidGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
};

class SigmoidDoubleGradNode : public egr::GradNodeBase {
 public:
  SigmoidDoubleGradNode() : egr::GradNodeBase() {}
  SigmoidDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SigmoidDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SigmoidDoubleGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();
    fwd_grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SigmoidDoubleGradNode>(
        new SigmoidDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }
  void SetTensorWrapperfwd_grad_out(
      const paddle::experimental::Tensor& fwd_grad_out) {
    fwd_grad_out_ = egr::TensorWrapper(fwd_grad_out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;
  egr::TensorWrapper fwd_grad_out_;

  // Attributes
};

class SigmoidTripleGradNode : public egr::GradNodeBase {
 public:
  SigmoidTripleGradNode() : egr::GradNodeBase() {}
  SigmoidTripleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SigmoidTripleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SigmoidTripleGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();
    fwd_grad_out_.clear();
    grad_grad_x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SigmoidTripleGradNode>(
        new SigmoidTripleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }
  void SetTensorWrapperfwd_grad_out(
      const paddle::experimental::Tensor& fwd_grad_out) {
    fwd_grad_out_ = egr::TensorWrapper(fwd_grad_out, false);
  }
  void SetTensorWrappergrad_grad_x(
      const paddle::experimental::Tensor& grad_grad_x) {
    grad_grad_x_ = egr::TensorWrapper(grad_grad_x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;
  egr::TensorWrapper fwd_grad_out_;
  egr::TensorWrapper grad_grad_x_;

  // Attributes
};

class SiluGradNode : public egr::GradNodeBase {
 public:
  SiluGradNode() : egr::GradNodeBase() {}
  SiluGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SiluGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SiluGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SiluGradNode>(new SiluGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class SinGradNode : public egr::GradNodeBase {
 public:
  SinGradNode() : egr::GradNodeBase() {}
  SinGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SinGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SinGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SinGradNode>(new SinGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class SinDoubleGradNode : public egr::GradNodeBase {
 public:
  SinDoubleGradNode() : egr::GradNodeBase() {}
  SinDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SinDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SinDoubleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SinDoubleGradNode>(new SinDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappergrad_out(const paddle::experimental::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper grad_out_;

  // Attributes
};

class SinTripleGradNode : public egr::GradNodeBase {
 public:
  SinTripleGradNode() : egr::GradNodeBase() {}
  SinTripleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SinTripleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SinTripleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    grad_out_forward_.clear();
    grad_x_grad_forward_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SinTripleGradNode>(new SinTripleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappergrad_out_forward(
      const paddle::experimental::Tensor& grad_out_forward) {
    grad_out_forward_ = egr::TensorWrapper(grad_out_forward, false);
  }
  void SetTensorWrappergrad_x_grad_forward(
      const paddle::experimental::Tensor& grad_x_grad_forward) {
    grad_x_grad_forward_ = egr::TensorWrapper(grad_x_grad_forward, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper grad_out_forward_;
  egr::TensorWrapper grad_x_grad_forward_;

  // Attributes
};

class SinhGradNode : public egr::GradNodeBase {
 public:
  SinhGradNode() : egr::GradNodeBase() {}
  SinhGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SinhGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SinhGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SinhGradNode>(new SinhGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class SoftplusGradNode : public egr::GradNodeBase {
 public:
  SoftplusGradNode() : egr::GradNodeBase() {}
  SoftplusGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SoftplusGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SoftplusGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SoftplusGradNode>(new SoftplusGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributebeta(const float& beta) { beta_ = beta; }
  void SetAttributethreshold(const float& threshold) { threshold_ = threshold; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  float beta_;
  float threshold_;
};

class SoftshrinkGradNode : public egr::GradNodeBase {
 public:
  SoftshrinkGradNode() : egr::GradNodeBase() {}
  SoftshrinkGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SoftshrinkGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SoftshrinkGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SoftshrinkGradNode>(new SoftshrinkGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributethreshold(const float& threshold) { threshold_ = threshold; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  float threshold_;
};

class SoftsignGradNode : public egr::GradNodeBase {
 public:
  SoftsignGradNode() : egr::GradNodeBase() {}
  SoftsignGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SoftsignGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SoftsignGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SoftsignGradNode>(new SoftsignGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class SolveGradNode : public egr::GradNodeBase {
 public:
  SolveGradNode() : egr::GradNodeBase() {}
  SolveGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SolveGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SolveGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SolveGradNode>(new SolveGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;
  egr::TensorWrapper out_;

  // Attributes
};

class SqrtGradNode : public egr::GradNodeBase {
 public:
  SqrtGradNode() : egr::GradNodeBase() {}
  SqrtGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SqrtGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SqrtGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SqrtGradNode>(new SqrtGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
};

class SqrtDoubleGradNode : public egr::GradNodeBase {
 public:
  SqrtDoubleGradNode() : egr::GradNodeBase() {}
  SqrtDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SqrtDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SqrtDoubleGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();
    grad_x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SqrtDoubleGradNode>(new SqrtDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }
  void SetTensorWrappergrad_x(const paddle::experimental::Tensor& grad_x) {
    grad_x_ = egr::TensorWrapper(grad_x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;
  egr::TensorWrapper grad_x_;

  // Attributes
};

class SquareGradNode : public egr::GradNodeBase {
 public:
  SquareGradNode() : egr::GradNodeBase() {}
  SquareGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SquareGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SquareGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SquareGradNode>(new SquareGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class SquareDoubleGradNode : public egr::GradNodeBase {
 public:
  SquareDoubleGradNode() : egr::GradNodeBase() {}
  SquareDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SquareDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SquareDoubleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SquareDoubleGradNode>(new SquareDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappergrad_out(const paddle::experimental::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper grad_out_;

  // Attributes
};

class SqueezeGradNode : public egr::GradNodeBase {
 public:
  SqueezeGradNode() : egr::GradNodeBase() {}
  SqueezeGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SqueezeGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SqueezeGradNode"; }

  void ClearTensorWrappers() override {
    xshape_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SqueezeGradNode>(new SqueezeGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperxshape(const paddle::experimental::Tensor& xshape) {
    xshape_ = egr::TensorWrapper(xshape, false);
  }

  // SetAttributes
  void SetAttributeaxis(const paddle::experimental::IntArray& axis) {
    axis_ = axis;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper xshape_;

  // Attributes
  paddle::experimental::IntArray axis_;
};

class SqueezeDoubleGradNode : public egr::GradNodeBase {
 public:
  SqueezeDoubleGradNode() : egr::GradNodeBase() {}
  SqueezeDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SqueezeDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SqueezeDoubleGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SqueezeDoubleGradNode>(
        new SqueezeDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeaxis(const paddle::experimental::IntArray& axis) {
    axis_ = axis;
  }

 private:
  // TensorWrappers

  // Attributes
  paddle::experimental::IntArray axis_;
};

class SvdGradNode : public egr::GradNodeBase {
 public:
  SvdGradNode() : egr::GradNodeBase() {}
  SvdGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SvdGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SvdGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    u_.clear();
    vh_.clear();
    s_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SvdGradNode>(new SvdGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperu(const paddle::experimental::Tensor& u) {
    u_ = egr::TensorWrapper(u, false);
  }
  void SetTensorWrappervh(const paddle::experimental::Tensor& vh) {
    vh_ = egr::TensorWrapper(vh, false);
  }
  void SetTensorWrappers(const paddle::experimental::Tensor& s) {
    s_ = egr::TensorWrapper(s, false);
  }

  // SetAttributes
  void SetAttributefull_matrices(const bool& full_matrices) {
    full_matrices_ = full_matrices;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper u_;
  egr::TensorWrapper vh_;
  egr::TensorWrapper s_;

  // Attributes
  bool full_matrices_;
};

class TakeAlongAxisGradNode : public egr::GradNodeBase {
 public:
  TakeAlongAxisGradNode() : egr::GradNodeBase() {}
  TakeAlongAxisGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TakeAlongAxisGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TakeAlongAxisGradNode"; }

  void ClearTensorWrappers() override {
    arr_.clear();
    indices_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<TakeAlongAxisGradNode>(
        new TakeAlongAxisGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperarr(const paddle::experimental::Tensor& arr) {
    arr_ = egr::TensorWrapper(arr, false);
  }
  void SetTensorWrapperindices(const paddle::experimental::Tensor& indices) {
    indices_ = egr::TensorWrapper(indices, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper arr_;
  egr::TensorWrapper indices_;

  // Attributes
  int axis_;
};

class TanGradNode : public egr::GradNodeBase {
 public:
  TanGradNode() : egr::GradNodeBase() {}
  TanGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TanGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TanGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<TanGradNode>(new TanGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class TanhGradNode : public egr::GradNodeBase {
 public:
  TanhGradNode() : egr::GradNodeBase() {}
  TanhGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TanhGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TanhGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<TanhGradNode>(new TanhGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
};

class TanhDoubleGradNode : public egr::GradNodeBase {
 public:
  TanhDoubleGradNode() : egr::GradNodeBase() {}
  TanhDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TanhDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TanhDoubleGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<TanhDoubleGradNode>(new TanhDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }
  void SetTensorWrappergrad_out(const paddle::experimental::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;
  egr::TensorWrapper grad_out_;

  // Attributes
};

class TanhTripleGradNode : public egr::GradNodeBase {
 public:
  TanhTripleGradNode() : egr::GradNodeBase() {}
  TanhTripleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TanhTripleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TanhTripleGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();
    grad_out_forward_.clear();
    grad_x_grad_forward_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<TanhTripleGradNode>(new TanhTripleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }
  void SetTensorWrappergrad_out_forward(
      const paddle::experimental::Tensor& grad_out_forward) {
    grad_out_forward_ = egr::TensorWrapper(grad_out_forward, false);
  }
  void SetTensorWrappergrad_x_grad_forward(
      const paddle::experimental::Tensor& grad_x_grad_forward) {
    grad_x_grad_forward_ = egr::TensorWrapper(grad_x_grad_forward, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;
  egr::TensorWrapper grad_out_forward_;
  egr::TensorWrapper grad_x_grad_forward_;

  // Attributes
};

class TanhShrinkGradNode : public egr::GradNodeBase {
 public:
  TanhShrinkGradNode() : egr::GradNodeBase() {}
  TanhShrinkGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TanhShrinkGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TanhShrinkGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<TanhShrinkGradNode>(new TanhShrinkGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class ThresholdedReluGradNode : public egr::GradNodeBase {
 public:
  ThresholdedReluGradNode() : egr::GradNodeBase() {}
  ThresholdedReluGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ThresholdedReluGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ThresholdedReluGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<ThresholdedReluGradNode>(
        new ThresholdedReluGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributethreshold(const float& threshold) { threshold_ = threshold; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  float threshold_;
};

class TopkGradNode : public egr::GradNodeBase {
 public:
  TopkGradNode() : egr::GradNodeBase() {}
  TopkGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TopkGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TopkGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    indices_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<TopkGradNode>(new TopkGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperindices(const paddle::experimental::Tensor& indices) {
    indices_ = egr::TensorWrapper(indices, false);
  }

  // SetAttributes
  void SetAttributek(const paddle::experimental::Scalar& k) { k_ = k; }
  void SetAttributeaxis(const int& axis) { axis_ = axis; }
  void SetAttributelargest(const bool& largest) { largest_ = largest; }
  void SetAttributesorted(const bool& sorted) { sorted_ = sorted; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper indices_;

  // Attributes
  paddle::experimental::Scalar k_;
  int axis_;
  bool largest_;
  bool sorted_;
};

class TraceGradNode : public egr::GradNodeBase {
 public:
  TraceGradNode() : egr::GradNodeBase() {}
  TraceGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TraceGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TraceGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<TraceGradNode>(new TraceGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }

  // SetAttributes
  void SetAttributeoffset(const int& offset) { offset_ = offset; }
  void SetAttributeaxis1(const int& axis1) { axis1_ = axis1; }
  void SetAttributeaxis2(const int& axis2) { axis2_ = axis2; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  int offset_;
  int axis1_;
  int axis2_;
};

class TruncGradNode : public egr::GradNodeBase {
 public:
  TruncGradNode() : egr::GradNodeBase() {}
  TruncGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TruncGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TruncGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<TruncGradNode>(new TruncGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes

 private:
  // TensorWrappers

  // Attributes
};

class UnfoldGradNode : public egr::GradNodeBase {
 public:
  UnfoldGradNode() : egr::GradNodeBase() {}
  UnfoldGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~UnfoldGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "UnfoldGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<UnfoldGradNode>(new UnfoldGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }

  // SetAttributes
  void SetAttributekernel_sizes(const std::vector<int>& kernel_sizes) {
    kernel_sizes_ = kernel_sizes;
  }
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributedilations(const std::vector<int>& dilations) {
    dilations_ = dilations;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  std::vector<int> kernel_sizes_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<int> dilations_;
};

class UnsqueezeGradNode : public egr::GradNodeBase {
 public:
  UnsqueezeGradNode() : egr::GradNodeBase() {}
  UnsqueezeGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~UnsqueezeGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "UnsqueezeGradNode"; }

  void ClearTensorWrappers() override {
    xshape_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<UnsqueezeGradNode>(new UnsqueezeGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperxshape(const paddle::experimental::Tensor& xshape) {
    xshape_ = egr::TensorWrapper(xshape, false);
  }

  // SetAttributes
  void SetAttributeaxes(const paddle::experimental::IntArray& axes) {
    axes_ = axes;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper xshape_;

  // Attributes
  paddle::experimental::IntArray axes_;
};

class UnsqueezeDoubleGradNode : public egr::GradNodeBase {
 public:
  UnsqueezeDoubleGradNode() : egr::GradNodeBase() {}
  UnsqueezeDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~UnsqueezeDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "UnsqueezeDoubleGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<UnsqueezeDoubleGradNode>(
        new UnsqueezeDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeaxes(const paddle::experimental::IntArray& axes) {
    axes_ = axes;
  }

 private:
  // TensorWrappers

  // Attributes
  paddle::experimental::IntArray axes_;
};

class UnstackGradNode : public egr::GradNodeBase {
 public:
  UnstackGradNode() : egr::GradNodeBase() {}
  UnstackGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~UnstackGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "UnstackGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<UnstackGradNode>(new UnstackGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers

  // Attributes
  int axis_;
};

class WarprnntGradNode : public egr::GradNodeBase {
 public:
  WarprnntGradNode() : egr::GradNodeBase() {}
  WarprnntGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~WarprnntGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "WarprnntGradNode"; }

  void ClearTensorWrappers() override {
    input_.clear();
    input_lengths_.clear();
    warprnntgrad_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<WarprnntGradNode>(new WarprnntGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperinput(const paddle::experimental::Tensor& input) {
    input_ = egr::TensorWrapper(input, true);
  }
  void SetTensorWrapperinput_lengths(
      const paddle::experimental::Tensor& input_lengths) {
    input_lengths_ = egr::TensorWrapper(input_lengths, false);
  }
  void SetTensorWrapperwarprnntgrad(
      const paddle::experimental::Tensor& warprnntgrad) {
    warprnntgrad_ = egr::TensorWrapper(warprnntgrad, false);
  }

  // SetAttributes
  void SetAttributeblank(const int& blank) { blank_ = blank; }
  void SetAttributefastemit_lambda(const float& fastemit_lambda) {
    fastemit_lambda_ = fastemit_lambda;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;
  egr::TensorWrapper input_lengths_;
  egr::TensorWrapper warprnntgrad_;

  // Attributes
  int blank_ = 0;
  float fastemit_lambda_ = 0.0;
};

class WhereGradNode : public egr::GradNodeBase {
 public:
  WhereGradNode() : egr::GradNodeBase() {}
  WhereGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~WhereGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "WhereGradNode"; }

  void ClearTensorWrappers() override {
    condition_.clear();
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<WhereGradNode>(new WhereGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrappercondition(
      const paddle::experimental::Tensor& condition) {
    condition_ = egr::TensorWrapper(condition, false);
  }
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, true);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper condition_;
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
};

class AbsGradNode : public egr::GradNodeBase {
 public:
  AbsGradNode() : egr::GradNodeBase() {}
  AbsGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AbsGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AbsGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AbsGradNode>(new AbsGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class AbsDoubleGradNode : public egr::GradNodeBase {
 public:
  AbsDoubleGradNode() : egr::GradNodeBase() {}
  AbsDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AbsDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AbsDoubleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<AbsDoubleGradNode>(new AbsDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class AddGradNode : public egr::GradNodeBase {
 public:
  AddGradNode() : egr::GradNodeBase() {}
  AddGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AddGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AddGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AddGradNode>(new AddGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, true);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
  int axis_ = -1;
};

class AddDoubleGradNode : public egr::GradNodeBase {
 public:
  AddDoubleGradNode() : egr::GradNodeBase() {}
  AddDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AddDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AddDoubleGradNode"; }

  void ClearTensorWrappers() override {
    y_.clear();
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<AddDoubleGradNode>(new AddDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }
  void SetTensorWrappergrad_out(const paddle::experimental::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper y_;
  egr::TensorWrapper grad_out_;

  // Attributes
  int axis_ = -1;
};

class AddTripleGradNode : public egr::GradNodeBase {
 public:
  AddTripleGradNode() : egr::GradNodeBase() {}
  AddTripleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AddTripleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AddTripleGradNode"; }

  void ClearTensorWrappers() override {
    grad_grad_x_.clear();
    grad_grad_y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<AddTripleGradNode>(new AddTripleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrappergrad_grad_x(
      const paddle::experimental::Tensor& grad_grad_x) {
    grad_grad_x_ = egr::TensorWrapper(grad_grad_x, false);
  }
  void SetTensorWrappergrad_grad_y(
      const paddle::experimental::Tensor& grad_grad_y) {
    grad_grad_y_ = egr::TensorWrapper(grad_grad_y, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper grad_grad_x_;
  egr::TensorWrapper grad_grad_y_;

  // Attributes
  int axis_ = -1;
};

class AffineGridGradNode : public egr::GradNodeBase {
 public:
  AffineGridGradNode() : egr::GradNodeBase() {}
  AffineGridGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AffineGridGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AffineGridGradNode"; }

  void ClearTensorWrappers() override {
    input_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<AffineGridGradNode>(new AffineGridGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperinput(const paddle::experimental::Tensor& input) {
    input_ = egr::TensorWrapper(input, true);
  }

  // SetAttributes
  void SetAttributeoutputShape(
      const paddle::experimental::IntArray& outputShape) {
    outputShape_ = outputShape;
  }
  void SetAttributealign_corners(const bool& align_corners) {
    align_corners_ = align_corners;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;

  // Attributes
  paddle::experimental::IntArray outputShape_;
  bool align_corners_ = true;
};

class AmaxGradNode : public egr::GradNodeBase {
 public:
  AmaxGradNode() : egr::GradNodeBase() {}
  AmaxGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AmaxGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AmaxGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AmaxGradNode>(new AmaxGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributeaxis(const std::vector<int64_t>& axis) { axis_ = axis; }
  void SetAttributekeepdim(const bool& keepdim) { keepdim_ = keepdim; }
  void SetAttributereduce_all(const bool& reduce_all) {
    reduce_all_ = reduce_all;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_;

  // Attributes
  std::vector<int64_t> axis_ = {};
  bool keepdim_ = false;
  bool reduce_all_ = false;
};

class AminGradNode : public egr::GradNodeBase {
 public:
  AminGradNode() : egr::GradNodeBase() {}
  AminGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AminGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AminGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AminGradNode>(new AminGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributeaxis(const std::vector<int64_t>& axis) { axis_ = axis; }
  void SetAttributekeepdim(const bool& keepdim) { keepdim_ = keepdim; }
  void SetAttributereduce_all(const bool& reduce_all) {
    reduce_all_ = reduce_all;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_;

  // Attributes
  std::vector<int64_t> axis_ = {};
  bool keepdim_ = false;
  bool reduce_all_ = false;
};

class AssignGradNode : public egr::GradNodeBase {
 public:
  AssignGradNode() : egr::GradNodeBase() {}
  AssignGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AssignGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AssignGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<AssignGradNode>(new AssignGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes

 private:
  // TensorWrappers

  // Attributes
};

class AssignOutGradNode : public egr::GradNodeBase {
 public:
  AssignOutGradNode() : egr::GradNodeBase() {}
  AssignOutGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AssignOutGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AssignOutGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<AssignOutGradNode>(new AssignOutGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes

 private:
  // TensorWrappers

  // Attributes
};

class BatchNormGradNode : public egr::GradNodeBase {
 public:
  BatchNormGradNode() : egr::GradNodeBase() {}
  BatchNormGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~BatchNormGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "BatchNormGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    scale_.clear();
    bias_.clear();
    mean_out_.clear();
    variance_out_.clear();
    saved_mean_.clear();
    saved_variance_.clear();
    reserve_space_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<BatchNormGradNode>(new BatchNormGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperscale(const paddle::experimental::Tensor& scale) {
    scale_ = egr::TensorWrapper(scale, false);
  }
  void SetTensorWrapperbias(const paddle::experimental::Tensor& bias) {
    bias_ = egr::TensorWrapper(bias, false);
  }
  void SetTensorWrappermean_out(const paddle::experimental::Tensor& mean_out) {
    mean_out_ = egr::TensorWrapper(mean_out, false);
  }
  void SetTensorWrappervariance_out(
      const paddle::experimental::Tensor& variance_out) {
    variance_out_ = egr::TensorWrapper(variance_out, false);
  }
  void SetTensorWrappersaved_mean(
      const paddle::experimental::Tensor& saved_mean) {
    saved_mean_ = egr::TensorWrapper(saved_mean, false);
  }
  void SetTensorWrappersaved_variance(
      const paddle::experimental::Tensor& saved_variance) {
    saved_variance_ = egr::TensorWrapper(saved_variance, false);
  }
  void SetTensorWrapperreserve_space(
      const paddle::experimental::Tensor& reserve_space) {
    reserve_space_ = egr::TensorWrapper(reserve_space, false);
  }

  // SetAttributes
  void SetAttributemomentum(const float& momentum) { momentum_ = momentum; }
  void SetAttributeepsilon(const float& epsilon) { epsilon_ = epsilon; }
  void SetAttributedata_layout(const std::string& data_layout) {
    data_layout_ = data_layout;
  }
  void SetAttributeis_test(const bool& is_test) { is_test_ = is_test; }
  void SetAttributeuse_global_stats(const bool& use_global_stats) {
    use_global_stats_ = use_global_stats;
  }
  void SetAttributetrainable_statistics(const bool& trainable_statistics) {
    trainable_statistics_ = trainable_statistics;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper scale_;
  egr::TensorWrapper bias_;
  egr::TensorWrapper mean_out_;
  egr::TensorWrapper variance_out_;
  egr::TensorWrapper saved_mean_;
  egr::TensorWrapper saved_variance_;
  egr::TensorWrapper reserve_space_;

  // Attributes
  float momentum_;
  float epsilon_;
  std::string data_layout_;
  bool is_test_;
  bool use_global_stats_;
  bool trainable_statistics_;
};

class BatchNormDoubleGradNode : public egr::GradNodeBase {
 public:
  BatchNormDoubleGradNode() : egr::GradNodeBase() {}
  BatchNormDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~BatchNormDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "BatchNormDoubleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    scale_.clear();
    out_mean_.clear();
    out_variance_.clear();
    saved_mean_.clear();
    saved_variance_.clear();
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<BatchNormDoubleGradNode>(
        new BatchNormDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperscale(const paddle::experimental::Tensor& scale) {
    scale_ = egr::TensorWrapper(scale, false);
  }
  void SetTensorWrapperout_mean(const paddle::experimental::Tensor& out_mean) {
    out_mean_ = egr::TensorWrapper(out_mean, false);
  }
  void SetTensorWrapperout_variance(
      const paddle::experimental::Tensor& out_variance) {
    out_variance_ = egr::TensorWrapper(out_variance, false);
  }
  void SetTensorWrappersaved_mean(
      const paddle::experimental::Tensor& saved_mean) {
    saved_mean_ = egr::TensorWrapper(saved_mean, false);
  }
  void SetTensorWrappersaved_variance(
      const paddle::experimental::Tensor& saved_variance) {
    saved_variance_ = egr::TensorWrapper(saved_variance, false);
  }
  void SetTensorWrappergrad_out(const paddle::experimental::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }

  // SetAttributes
  void SetAttributemomentum(const float& momentum) { momentum_ = momentum; }
  void SetAttributeepsilon(const float& epsilon) { epsilon_ = epsilon; }
  void SetAttributedata_layout(const std::string& data_layout) {
    data_layout_ = data_layout;
  }
  void SetAttributeis_test(const bool& is_test) { is_test_ = is_test; }
  void SetAttributeuse_global_stats(const bool& use_global_stats) {
    use_global_stats_ = use_global_stats;
  }
  void SetAttributetrainable_statistics(const bool& trainable_statistics) {
    trainable_statistics_ = trainable_statistics;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper scale_;
  egr::TensorWrapper out_mean_;
  egr::TensorWrapper out_variance_;
  egr::TensorWrapper saved_mean_;
  egr::TensorWrapper saved_variance_;
  egr::TensorWrapper grad_out_;

  // Attributes
  float momentum_;
  float epsilon_;
  std::string data_layout_;
  bool is_test_;
  bool use_global_stats_;
  bool trainable_statistics_;
};

class BceLossGradNode : public egr::GradNodeBase {
 public:
  BceLossGradNode() : egr::GradNodeBase() {}
  BceLossGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~BceLossGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "BceLossGradNode"; }

  void ClearTensorWrappers() override {
    input_.clear();
    label_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<BceLossGradNode>(new BceLossGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperinput(const paddle::experimental::Tensor& input) {
    input_ = egr::TensorWrapper(input, false);
  }
  void SetTensorWrapperlabel(const paddle::experimental::Tensor& label) {
    label_ = egr::TensorWrapper(label, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper input_;
  egr::TensorWrapper label_;

  // Attributes
};

class BicubicInterpGradNode : public egr::GradNodeBase {
 public:
  BicubicInterpGradNode() : egr::GradNodeBase() {}
  BicubicInterpGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~BicubicInterpGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "BicubicInterpGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_size_.clear();
    for (auto& tw : size_tensor_) {
      tw.clear();
    }
    scale_tensor_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<BicubicInterpGradNode>(
        new BicubicInterpGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout_size(const paddle::experimental::Tensor& out_size) {
    out_size_ = egr::TensorWrapper(out_size, false);
  }
  void SetTensorWrappersize_tensor(
      const std::vector<paddle::experimental::Tensor>& size_tensor) {
    for (const auto& eager_tensor : size_tensor) {
      size_tensor_.emplace_back(egr::TensorWrapper(eager_tensor, false));
    };
  }
  void SetTensorWrapperscale_tensor(
      const paddle::experimental::Tensor& scale_tensor) {
    scale_tensor_ = egr::TensorWrapper(scale_tensor, false);
  }

  // SetAttributes
  void SetAttributedata_layout(const std::string& data_layout) {
    data_layout_ = data_layout;
  }
  void SetAttributeout_d(const int& out_d) { out_d_ = out_d; }
  void SetAttributeout_h(const int& out_h) { out_h_ = out_h; }
  void SetAttributeout_w(const int& out_w) { out_w_ = out_w; }
  void SetAttributescale(const std::vector<float>& scale) { scale_ = scale; }
  void SetAttributeinterp_method(const std::string& interp_method) {
    interp_method_ = interp_method;
  }
  void SetAttributealign_corners(const bool& align_corners) {
    align_corners_ = align_corners;
  }
  void SetAttributealign_mode(const int& align_mode) {
    align_mode_ = align_mode;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_size_;
  std::vector<egr::TensorWrapper> size_tensor_;
  egr::TensorWrapper scale_tensor_;

  // Attributes
  std::string data_layout_;
  int out_d_;
  int out_h_;
  int out_w_;
  std::vector<float> scale_;
  std::string interp_method_;
  bool align_corners_;
  int align_mode_;
};

class BilinearInterpGradNode : public egr::GradNodeBase {
 public:
  BilinearInterpGradNode() : egr::GradNodeBase() {}
  BilinearInterpGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~BilinearInterpGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "BilinearInterpGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_size_.clear();
    for (auto& tw : size_tensor_) {
      tw.clear();
    }
    scale_tensor_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<BilinearInterpGradNode>(
        new BilinearInterpGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout_size(const paddle::experimental::Tensor& out_size) {
    out_size_ = egr::TensorWrapper(out_size, false);
  }
  void SetTensorWrappersize_tensor(
      const std::vector<paddle::experimental::Tensor>& size_tensor) {
    for (const auto& eager_tensor : size_tensor) {
      size_tensor_.emplace_back(egr::TensorWrapper(eager_tensor, false));
    };
  }
  void SetTensorWrapperscale_tensor(
      const paddle::experimental::Tensor& scale_tensor) {
    scale_tensor_ = egr::TensorWrapper(scale_tensor, false);
  }

  // SetAttributes
  void SetAttributedata_layout(const std::string& data_layout) {
    data_layout_ = data_layout;
  }
  void SetAttributeout_d(const int& out_d) { out_d_ = out_d; }
  void SetAttributeout_h(const int& out_h) { out_h_ = out_h; }
  void SetAttributeout_w(const int& out_w) { out_w_ = out_w; }
  void SetAttributescale(const std::vector<float>& scale) { scale_ = scale; }
  void SetAttributeinterp_method(const std::string& interp_method) {
    interp_method_ = interp_method;
  }
  void SetAttributealign_corners(const bool& align_corners) {
    align_corners_ = align_corners;
  }
  void SetAttributealign_mode(const int& align_mode) {
    align_mode_ = align_mode;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_size_;
  std::vector<egr::TensorWrapper> size_tensor_;
  egr::TensorWrapper scale_tensor_;

  // Attributes
  std::string data_layout_;
  int out_d_;
  int out_h_;
  int out_w_;
  std::vector<float> scale_;
  std::string interp_method_;
  bool align_corners_;
  int align_mode_;
};

class BilinearTensorProductGradNode : public egr::GradNodeBase {
 public:
  BilinearTensorProductGradNode() : egr::GradNodeBase() {}
  BilinearTensorProductGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~BilinearTensorProductGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "BilinearTensorProductGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();
    weight_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<BilinearTensorProductGradNode>(
        new BilinearTensorProductGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }
  void SetTensorWrapperweight(const paddle::experimental::Tensor& weight) {
    weight_ = egr::TensorWrapper(weight, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;
  egr::TensorWrapper weight_;

  // Attributes
};

class BroadcastTensorsGradNode : public egr::GradNodeBase {
 public:
  BroadcastTensorsGradNode() : egr::GradNodeBase() {}
  BroadcastTensorsGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~BroadcastTensorsGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "BroadcastTensorsGradNode"; }

  void ClearTensorWrappers() override {
    for (auto& tw : input_) {
      tw.clear();
    }

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<BroadcastTensorsGradNode>(
        new BroadcastTensorsGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperinput(
      const std::vector<paddle::experimental::Tensor>& input) {
    for (const auto& eager_tensor : input) {
      input_.emplace_back(egr::TensorWrapper(eager_tensor, true));
    };
  }

  // SetAttributes

 private:
  // TensorWrappers
  std::vector<egr::TensorWrapper> input_;

  // Attributes
};

class CastGradNode : public egr::GradNodeBase {
 public:
  CastGradNode() : egr::GradNodeBase() {}
  CastGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~CastGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "CastGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<CastGradNode>(new CastGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class ConcatGradNode : public egr::GradNodeBase {
 public:
  ConcatGradNode() : egr::GradNodeBase() {}
  ConcatGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ConcatGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ConcatGradNode"; }

  void ClearTensorWrappers() override {
    for (auto& tw : x_) {
      tw.clear();
    }

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ConcatGradNode>(new ConcatGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const std::vector<paddle::experimental::Tensor>& x) {
    for (const auto& eager_tensor : x) {
      x_.emplace_back(egr::TensorWrapper(eager_tensor, true));
    };
  }

  // SetAttributes
  void SetAttributeaxis(const paddle::experimental::Scalar& axis) {
    axis_ = axis;
  }

 private:
  // TensorWrappers
  std::vector<egr::TensorWrapper> x_;

  // Attributes
  paddle::experimental::Scalar axis_ = 0;
};

class ConcatDoubleGradNode : public egr::GradNodeBase {
 public:
  ConcatDoubleGradNode() : egr::GradNodeBase() {}
  ConcatDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ConcatDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ConcatDoubleGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ConcatDoubleGradNode>(new ConcatDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeaxis(const paddle::experimental::Scalar& axis) {
    axis_ = axis;
  }

 private:
  // TensorWrappers

  // Attributes
  paddle::experimental::Scalar axis_ = 0;
};

class Conv2dTransposeGradNode : public egr::GradNodeBase {
 public:
  Conv2dTransposeGradNode() : egr::GradNodeBase() {}
  Conv2dTransposeGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Conv2dTransposeGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Conv2dTransposeGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    filter_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<Conv2dTransposeGradNode>(
        new Conv2dTransposeGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperfilter(const paddle::experimental::Tensor& filter) {
    filter_ = egr::TensorWrapper(filter, false);
  }

  // SetAttributes
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributeoutput_padding(const std::vector<int>& output_padding) {
    output_padding_ = output_padding;
  }
  void SetAttributeoutput_size(
      const paddle::experimental::IntArray& output_size) {
    output_size_ = output_size;
  }
  void SetAttributepadding_algorithm(const std::string& padding_algorithm) {
    padding_algorithm_ = padding_algorithm;
  }
  void SetAttributegroups(const int& groups) { groups_ = groups; }
  void SetAttributedilations(const std::vector<int>& dilations) {
    dilations_ = dilations;
  }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper filter_;

  // Attributes
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<int> output_padding_;
  paddle::experimental::IntArray output_size_;
  std::string padding_algorithm_;
  int groups_;
  std::vector<int> dilations_;
  std::string data_format_;
};

class Conv2dTransposeDoubleGradNode : public egr::GradNodeBase {
 public:
  Conv2dTransposeDoubleGradNode() : egr::GradNodeBase() {}
  Conv2dTransposeDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Conv2dTransposeDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Conv2dTransposeDoubleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    filter_.clear();
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<Conv2dTransposeDoubleGradNode>(
        new Conv2dTransposeDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperfilter(const paddle::experimental::Tensor& filter) {
    filter_ = egr::TensorWrapper(filter, false);
  }
  void SetTensorWrappergrad_out(const paddle::experimental::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }

  // SetAttributes
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributeoutput_padding(const std::vector<int>& output_padding) {
    output_padding_ = output_padding;
  }
  void SetAttributeoutput_size(
      const paddle::experimental::IntArray& output_size) {
    output_size_ = output_size;
  }
  void SetAttributepadding_algorithm(const std::string& padding_algorithm) {
    padding_algorithm_ = padding_algorithm;
  }
  void SetAttributegroups(const int& groups) { groups_ = groups; }
  void SetAttributedilations(const std::vector<int>& dilations) {
    dilations_ = dilations;
  }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper filter_;
  egr::TensorWrapper grad_out_;

  // Attributes
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<int> output_padding_;
  paddle::experimental::IntArray output_size_;
  std::string padding_algorithm_;
  int groups_;
  std::vector<int> dilations_;
  std::string data_format_;
};

class Conv3dGradNode : public egr::GradNodeBase {
 public:
  Conv3dGradNode() : egr::GradNodeBase() {}
  Conv3dGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Conv3dGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Conv3dGradNode"; }

  void ClearTensorWrappers() override {
    input_.clear();
    filter_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<Conv3dGradNode>(new Conv3dGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperinput(const paddle::experimental::Tensor& input) {
    input_ = egr::TensorWrapper(input, false);
  }
  void SetTensorWrapperfilter(const paddle::experimental::Tensor& filter) {
    filter_ = egr::TensorWrapper(filter, false);
  }

  // SetAttributes
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributepadding_algorithm(const std::string& padding_algorithm) {
    padding_algorithm_ = padding_algorithm;
  }
  void SetAttributegroups(const int& groups) { groups_ = groups; }
  void SetAttributedilations(const std::vector<int>& dilations) {
    dilations_ = dilations;
  }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;
  egr::TensorWrapper filter_;

  // Attributes
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::string padding_algorithm_;
  int groups_;
  std::vector<int> dilations_;
  std::string data_format_;
};

class Conv3dDoubleGradNode : public egr::GradNodeBase {
 public:
  Conv3dDoubleGradNode() : egr::GradNodeBase() {}
  Conv3dDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Conv3dDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Conv3dDoubleGradNode"; }

  void ClearTensorWrappers() override {
    input_.clear();
    filter_.clear();
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<Conv3dDoubleGradNode>(new Conv3dDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperinput(const paddle::experimental::Tensor& input) {
    input_ = egr::TensorWrapper(input, false);
  }
  void SetTensorWrapperfilter(const paddle::experimental::Tensor& filter) {
    filter_ = egr::TensorWrapper(filter, false);
  }
  void SetTensorWrappergrad_out(const paddle::experimental::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }

  // SetAttributes
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributepadding_algorithm(const std::string& padding_algorithm) {
    padding_algorithm_ = padding_algorithm;
  }
  void SetAttributegroups(const int& groups) { groups_ = groups; }
  void SetAttributedilations(const std::vector<int>& dilations) {
    dilations_ = dilations;
  }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;
  egr::TensorWrapper filter_;
  egr::TensorWrapper grad_out_;

  // Attributes
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::string padding_algorithm_;
  int groups_;
  std::vector<int> dilations_;
  std::string data_format_;
};

class Conv3dTransposeGradNode : public egr::GradNodeBase {
 public:
  Conv3dTransposeGradNode() : egr::GradNodeBase() {}
  Conv3dTransposeGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Conv3dTransposeGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Conv3dTransposeGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    filter_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<Conv3dTransposeGradNode>(
        new Conv3dTransposeGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperfilter(const paddle::experimental::Tensor& filter) {
    filter_ = egr::TensorWrapper(filter, false);
  }

  // SetAttributes
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributeoutput_padding(const std::vector<int>& output_padding) {
    output_padding_ = output_padding;
  }
  void SetAttributeoutput_size(const std::vector<int>& output_size) {
    output_size_ = output_size;
  }
  void SetAttributepadding_algorithm(const std::string& padding_algorithm) {
    padding_algorithm_ = padding_algorithm;
  }
  void SetAttributegroups(const int& groups) { groups_ = groups; }
  void SetAttributedilations(const std::vector<int>& dilations) {
    dilations_ = dilations;
  }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper filter_;

  // Attributes
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<int> output_padding_;
  std::vector<int> output_size_;
  std::string padding_algorithm_;
  int groups_;
  std::vector<int> dilations_;
  std::string data_format_;
};

class CrossEntropyWithSoftmaxGradNode : public egr::GradNodeBase {
 public:
  CrossEntropyWithSoftmaxGradNode() : egr::GradNodeBase() {}
  CrossEntropyWithSoftmaxGradNode(size_t bwd_in_slot_num,
                                  size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~CrossEntropyWithSoftmaxGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "CrossEntropyWithSoftmaxGradNode"; }

  void ClearTensorWrappers() override {
    label_.clear();
    softmax_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<CrossEntropyWithSoftmaxGradNode>(
        new CrossEntropyWithSoftmaxGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperlabel(const paddle::experimental::Tensor& label) {
    label_ = egr::TensorWrapper(label, false);
  }
  void SetTensorWrappersoftmax(const paddle::experimental::Tensor& softmax) {
    softmax_ = egr::TensorWrapper(softmax, false);
  }

  // SetAttributes
  void SetAttributesoft_label(const bool& soft_label) {
    soft_label_ = soft_label;
  }
  void SetAttributeuse_softmax(const bool& use_softmax) {
    use_softmax_ = use_softmax;
  }
  void SetAttributenumeric_stable_mode(const bool& numeric_stable_mode) {
    numeric_stable_mode_ = numeric_stable_mode;
  }
  void SetAttributeignore_index(const int& ignore_index) {
    ignore_index_ = ignore_index;
  }
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper label_;
  egr::TensorWrapper softmax_;

  // Attributes
  bool soft_label_;
  bool use_softmax_;
  bool numeric_stable_mode_;
  int ignore_index_;
  int axis_;
};

class CumprodGradNode : public egr::GradNodeBase {
 public:
  CumprodGradNode() : egr::GradNodeBase() {}
  CumprodGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~CumprodGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "CumprodGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<CumprodGradNode>(new CumprodGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributedim(const int& dim) { dim_ = dim; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_;

  // Attributes
  int dim_;
};

class CumsumGradNode : public egr::GradNodeBase {
 public:
  CumsumGradNode() : egr::GradNodeBase() {}
  CumsumGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~CumsumGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "CumsumGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<CumsumGradNode>(new CumsumGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeaxis(const paddle::experimental::Scalar& axis) {
    axis_ = axis;
  }
  void SetAttributeflatten(const bool& flatten) { flatten_ = flatten; }
  void SetAttributeexclusive(const bool& exclusive) { exclusive_ = exclusive; }
  void SetAttributereverse(const bool& reverse) { reverse_ = reverse; }

 private:
  // TensorWrappers

  // Attributes
  paddle::experimental::Scalar axis_;
  bool flatten_;
  bool exclusive_;
  bool reverse_;
};

class DeformableConvGradNode : public egr::GradNodeBase {
 public:
  DeformableConvGradNode() : egr::GradNodeBase() {}
  DeformableConvGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~DeformableConvGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "DeformableConvGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    offset_.clear();
    filter_.clear();
    mask_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<DeformableConvGradNode>(
        new DeformableConvGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperoffset(const paddle::experimental::Tensor& offset) {
    offset_ = egr::TensorWrapper(offset, false);
  }
  void SetTensorWrapperfilter(const paddle::experimental::Tensor& filter) {
    filter_ = egr::TensorWrapper(filter, false);
  }
  void SetTensorWrappermask(const paddle::experimental::Tensor& mask) {
    mask_ = egr::TensorWrapper(mask, false);
  }

  // SetAttributes
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributedilations(const std::vector<int>& dilations) {
    dilations_ = dilations;
  }
  void SetAttributedeformable_groups(const int& deformable_groups) {
    deformable_groups_ = deformable_groups;
  }
  void SetAttributegroups(const int& groups) { groups_ = groups; }
  void SetAttributeim2col_step(const int& im2col_step) {
    im2col_step_ = im2col_step;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper offset_;
  egr::TensorWrapper filter_;
  egr::TensorWrapper mask_;

  // Attributes
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<int> dilations_;
  int deformable_groups_;
  int groups_;
  int im2col_step_;
};

class DepthwiseConv2dGradNode : public egr::GradNodeBase {
 public:
  DepthwiseConv2dGradNode() : egr::GradNodeBase() {}
  DepthwiseConv2dGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~DepthwiseConv2dGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "DepthwiseConv2dGradNode"; }

  void ClearTensorWrappers() override {
    input_.clear();
    filter_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<DepthwiseConv2dGradNode>(
        new DepthwiseConv2dGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperinput(const paddle::experimental::Tensor& input) {
    input_ = egr::TensorWrapper(input, false);
  }
  void SetTensorWrapperfilter(const paddle::experimental::Tensor& filter) {
    filter_ = egr::TensorWrapper(filter, false);
  }

  // SetAttributes
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributepadding_algorithm(const std::string& padding_algorithm) {
    padding_algorithm_ = padding_algorithm;
  }
  void SetAttributegroups(const int& groups) { groups_ = groups; }
  void SetAttributedilations(const std::vector<int>& dilations) {
    dilations_ = dilations;
  }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;
  egr::TensorWrapper filter_;

  // Attributes
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::string padding_algorithm_;
  int groups_;
  std::vector<int> dilations_;
  std::string data_format_;
};

class DepthwiseConv2dDoubleGradNode : public egr::GradNodeBase {
 public:
  DepthwiseConv2dDoubleGradNode() : egr::GradNodeBase() {}
  DepthwiseConv2dDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~DepthwiseConv2dDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "DepthwiseConv2dDoubleGradNode"; }

  void ClearTensorWrappers() override {
    input_.clear();
    filter_.clear();
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<DepthwiseConv2dDoubleGradNode>(
        new DepthwiseConv2dDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperinput(const paddle::experimental::Tensor& input) {
    input_ = egr::TensorWrapper(input, false);
  }
  void SetTensorWrapperfilter(const paddle::experimental::Tensor& filter) {
    filter_ = egr::TensorWrapper(filter, false);
  }
  void SetTensorWrappergrad_out(const paddle::experimental::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }

  // SetAttributes
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributepadding_algorithm(const std::string& padding_algorithm) {
    padding_algorithm_ = padding_algorithm;
  }
  void SetAttributegroups(const int& groups) { groups_ = groups; }
  void SetAttributedilations(const std::vector<int>& dilations) {
    dilations_ = dilations;
  }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;
  egr::TensorWrapper filter_;
  egr::TensorWrapper grad_out_;

  // Attributes
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::string padding_algorithm_;
  int groups_;
  std::vector<int> dilations_;
  std::string data_format_;
};

class DepthwiseConv2dTransposeGradNode : public egr::GradNodeBase {
 public:
  DepthwiseConv2dTransposeGradNode() : egr::GradNodeBase() {}
  DepthwiseConv2dTransposeGradNode(size_t bwd_in_slot_num,
                                   size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~DepthwiseConv2dTransposeGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "DepthwiseConv2dTransposeGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    filter_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<DepthwiseConv2dTransposeGradNode>(
        new DepthwiseConv2dTransposeGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperfilter(const paddle::experimental::Tensor& filter) {
    filter_ = egr::TensorWrapper(filter, false);
  }

  // SetAttributes
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributeoutput_padding(const std::vector<int>& output_padding) {
    output_padding_ = output_padding;
  }
  void SetAttributeoutput_size(
      const paddle::experimental::IntArray& output_size) {
    output_size_ = output_size;
  }
  void SetAttributepadding_algorithm(const std::string& padding_algorithm) {
    padding_algorithm_ = padding_algorithm;
  }
  void SetAttributegroups(const int& groups) { groups_ = groups; }
  void SetAttributedilations(const std::vector<int>& dilations) {
    dilations_ = dilations;
  }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper filter_;

  // Attributes
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<int> output_padding_;
  paddle::experimental::IntArray output_size_;
  std::string padding_algorithm_;
  int groups_;
  std::vector<int> dilations_;
  std::string data_format_;
};

class DivideGradNode : public egr::GradNodeBase {
 public:
  DivideGradNode() : egr::GradNodeBase() {}
  DivideGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~DivideGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "DivideGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<DivideGradNode>(new DivideGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;
  egr::TensorWrapper out_;

  // Attributes
  int axis_ = -1;
};

class DivideDoubleGradNode : public egr::GradNodeBase {
 public:
  DivideDoubleGradNode() : egr::GradNodeBase() {}
  DivideDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~DivideDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "DivideDoubleGradNode"; }

  void ClearTensorWrappers() override {
    y_.clear();
    out_.clear();
    grad_x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<DivideDoubleGradNode>(new DivideDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }
  void SetTensorWrappergrad_x(const paddle::experimental::Tensor& grad_x) {
    grad_x_ = egr::TensorWrapper(grad_x, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper y_;
  egr::TensorWrapper out_;
  egr::TensorWrapper grad_x_;

  // Attributes
  int axis_ = -1;
};

class DropoutGradNode : public egr::GradNodeBase {
 public:
  DropoutGradNode() : egr::GradNodeBase() {}
  DropoutGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~DropoutGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "DropoutGradNode"; }

  void ClearTensorWrappers() override {
    mask_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<DropoutGradNode>(new DropoutGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrappermask(const paddle::experimental::Tensor& mask) {
    mask_ = egr::TensorWrapper(mask, false);
  }

  // SetAttributes
  void SetAttributep(const paddle::experimental::Scalar& p) { p_ = p; }
  void SetAttributeis_test(const bool& is_test) { is_test_ = is_test; }
  void SetAttributemode(const std::string& mode) { mode_ = mode; }

 private:
  // TensorWrappers
  egr::TensorWrapper mask_;

  // Attributes
  paddle::experimental::Scalar p_;
  bool is_test_;
  std::string mode_;
};

class EigvalshGradNode : public egr::GradNodeBase {
 public:
  EigvalshGradNode() : egr::GradNodeBase() {}
  EigvalshGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~EigvalshGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "EigvalshGradNode"; }

  void ClearTensorWrappers() override {
    eigenvectors_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<EigvalshGradNode>(new EigvalshGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrappereigenvectors(
      const paddle::experimental::Tensor& eigenvectors) {
    eigenvectors_ = egr::TensorWrapper(eigenvectors, false);
  }

  // SetAttributes
  void SetAttributeuplo(const std::string& uplo) { uplo_ = uplo; }
  void SetAttributeis_test(const bool& is_test) { is_test_ = is_test; }

 private:
  // TensorWrappers
  egr::TensorWrapper eigenvectors_;

  // Attributes
  std::string uplo_;
  bool is_test_;
};

class EinsumGradNode : public egr::GradNodeBase {
 public:
  EinsumGradNode() : egr::GradNodeBase() {}
  EinsumGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~EinsumGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "EinsumGradNode"; }

  void ClearTensorWrappers() override {
    for (auto& tw : x_shape_) {
      tw.clear();
    }
    for (auto& tw : inner_cache_) {
      tw.clear();
    }

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<EinsumGradNode>(new EinsumGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx_shape(
      const std::vector<paddle::experimental::Tensor>& x_shape) {
    for (const auto& eager_tensor : x_shape) {
      x_shape_.emplace_back(egr::TensorWrapper(eager_tensor, false));
    };
  }
  void SetTensorWrapperinner_cache(
      const std::vector<paddle::experimental::Tensor>& inner_cache) {
    for (const auto& eager_tensor : inner_cache) {
      inner_cache_.emplace_back(egr::TensorWrapper(eager_tensor, false));
    };
  }

  // SetAttributes
  void SetAttributeequation(const std::string& equation) {
    equation_ = equation;
  }

 private:
  // TensorWrappers
  std::vector<egr::TensorWrapper> x_shape_;
  std::vector<egr::TensorWrapper> inner_cache_;

  // Attributes
  std::string equation_;
};

class ElementwisePowGradNode : public egr::GradNodeBase {
 public:
  ElementwisePowGradNode() : egr::GradNodeBase() {}
  ElementwisePowGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ElementwisePowGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ElementwisePowGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<ElementwisePowGradNode>(
        new ElementwisePowGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
  int axis_ = -1;
};

class EmbeddingGradNode : public egr::GradNodeBase {
 public:
  EmbeddingGradNode() : egr::GradNodeBase() {}
  EmbeddingGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~EmbeddingGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "EmbeddingGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    weight_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<EmbeddingGradNode>(new EmbeddingGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperweight(const paddle::experimental::Tensor& weight) {
    weight_ = egr::TensorWrapper(weight, true);
  }

  // SetAttributes
  void SetAttributepadding_idx(const int64_t& padding_idx) {
    padding_idx_ = padding_idx;
  }
  void SetAttributesparse(const bool& sparse) { sparse_ = sparse; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper weight_;

  // Attributes
  int64_t padding_idx_ = -1;
  bool sparse_ = false;
};

class ExpandGradNode : public egr::GradNodeBase {
 public:
  ExpandGradNode() : egr::GradNodeBase() {}
  ExpandGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ExpandGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ExpandGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ExpandGradNode>(new ExpandGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }

  // SetAttributes
  void SetAttributeshape(const paddle::experimental::IntArray& shape) {
    shape_ = shape;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  paddle::experimental::IntArray shape_;
};

class ExpandDoubleGradNode : public egr::GradNodeBase {
 public:
  ExpandDoubleGradNode() : egr::GradNodeBase() {}
  ExpandDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ExpandDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ExpandDoubleGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ExpandDoubleGradNode>(new ExpandDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeshape(const paddle::experimental::IntArray& shape) {
    shape_ = shape;
  }

 private:
  // TensorWrappers

  // Attributes
  paddle::experimental::IntArray shape_;
};

class ExpandAsGradNode : public egr::GradNodeBase {
 public:
  ExpandAsGradNode() : egr::GradNodeBase() {}
  ExpandAsGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ExpandAsGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ExpandAsGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ExpandAsGradNode>(new ExpandAsGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }

  // SetAttributes
  void SetAttributetarget_shape(const std::vector<int>& target_shape) {
    target_shape_ = target_shape;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  std::vector<int> target_shape_;
};

class ExponentialGradNode : public egr::GradNodeBase {
 public:
  ExponentialGradNode() : egr::GradNodeBase() {}
  ExponentialGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ExponentialGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ExponentialGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ExponentialGradNode>(new ExponentialGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes

 private:
  // TensorWrappers

  // Attributes
};

class FillGradNode : public egr::GradNodeBase {
 public:
  FillGradNode() : egr::GradNodeBase() {}
  FillGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~FillGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "FillGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<FillGradNode>(new FillGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributevalue(const paddle::experimental::Scalar& value) {
    value_ = value;
  }

 private:
  // TensorWrappers

  // Attributes
  paddle::experimental::Scalar value_;
};

class FlattenGradNode : public egr::GradNodeBase {
 public:
  FlattenGradNode() : egr::GradNodeBase() {}
  FlattenGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~FlattenGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "FlattenGradNode"; }

  void ClearTensorWrappers() override {
    xshape_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<FlattenGradNode>(new FlattenGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperxshape(const paddle::experimental::Tensor& xshape) {
    xshape_ = egr::TensorWrapper(xshape, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper xshape_;

  // Attributes
};

class FmaxGradNode : public egr::GradNodeBase {
 public:
  FmaxGradNode() : egr::GradNodeBase() {}
  FmaxGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~FmaxGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "FmaxGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<FmaxGradNode>(new FmaxGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
  int axis_ = -1;
};

class FminGradNode : public egr::GradNodeBase {
 public:
  FminGradNode() : egr::GradNodeBase() {}
  FminGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~FminGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "FminGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<FminGradNode>(new FminGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
  int axis_ = -1;
};

class FrobeniusNormGradNode : public egr::GradNodeBase {
 public:
  FrobeniusNormGradNode() : egr::GradNodeBase() {}
  FrobeniusNormGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~FrobeniusNormGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "FrobeniusNormGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<FrobeniusNormGradNode>(
        new FrobeniusNormGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributeaxis(const std::vector<int64_t>& axis) { axis_ = axis; }
  void SetAttributekeep_dim(const bool& keep_dim) { keep_dim_ = keep_dim; }
  void SetAttributereduce_all(const bool& reduce_all) {
    reduce_all_ = reduce_all;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_;

  // Attributes
  std::vector<int64_t> axis_;
  bool keep_dim_;
  bool reduce_all_;
};

class GatherGradNode : public egr::GradNodeBase {
 public:
  GatherGradNode() : egr::GradNodeBase() {}
  GatherGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~GatherGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "GatherGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    index_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<GatherGradNode>(new GatherGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }
  void SetTensorWrapperindex(const paddle::experimental::Tensor& index) {
    index_ = egr::TensorWrapper(index, false);
  }

  // SetAttributes
  void SetAttributeaxis(const paddle::experimental::Scalar& axis) {
    axis_ = axis;
  }
  void SetAttributeoverwrite(const bool& overwrite) { overwrite_ = overwrite; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper index_;

  // Attributes
  paddle::experimental::Scalar axis_ = 0;
  bool overwrite_ = false;
};

class GroupNormGradNode : public egr::GradNodeBase {
 public:
  GroupNormGradNode() : egr::GradNodeBase() {}
  GroupNormGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~GroupNormGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "GroupNormGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    scale_.clear();
    bias_.clear();
    y_.clear();
    mean_.clear();
    variance_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<GroupNormGradNode>(new GroupNormGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperscale(const paddle::experimental::Tensor& scale) {
    scale_ = egr::TensorWrapper(scale, false);
  }
  void SetTensorWrapperbias(const paddle::experimental::Tensor& bias) {
    bias_ = egr::TensorWrapper(bias, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }
  void SetTensorWrappermean(const paddle::experimental::Tensor& mean) {
    mean_ = egr::TensorWrapper(mean, false);
  }
  void SetTensorWrappervariance(const paddle::experimental::Tensor& variance) {
    variance_ = egr::TensorWrapper(variance, false);
  }

  // SetAttributes
  void SetAttributeepsilon(const float& epsilon) { epsilon_ = epsilon; }
  void SetAttributegroups(const int& groups) { groups_ = groups; }
  void SetAttributedata_layout(const std::string& data_layout) {
    data_layout_ = data_layout;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper scale_;
  egr::TensorWrapper bias_;
  egr::TensorWrapper y_;
  egr::TensorWrapper mean_;
  egr::TensorWrapper variance_;

  // Attributes
  float epsilon_;
  int groups_;
  std::string data_layout_;
};

class HardswishGradNode : public egr::GradNodeBase {
 public:
  HardswishGradNode() : egr::GradNodeBase() {}
  HardswishGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~HardswishGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "HardswishGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<HardswishGradNode>(new HardswishGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributethreshold(const float& threshold) { threshold_ = threshold; }
  void SetAttributescale(const float& scale) { scale_ = scale; }
  void SetAttributeoffset(const float& offset) { offset_ = offset; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  float threshold_ = 6.0;
  float scale_ = 6.0;
  float offset_ = 3.0;
};

class HsigmoidLossGradNode : public egr::GradNodeBase {
 public:
  HsigmoidLossGradNode() : egr::GradNodeBase() {}
  HsigmoidLossGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~HsigmoidLossGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "HsigmoidLossGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    w_.clear();
    label_.clear();
    path_.clear();
    code_.clear();
    bias_.clear();
    pre_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<HsigmoidLossGradNode>(new HsigmoidLossGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperw(const paddle::experimental::Tensor& w) {
    w_ = egr::TensorWrapper(w, false);
  }
  void SetTensorWrapperlabel(const paddle::experimental::Tensor& label) {
    label_ = egr::TensorWrapper(label, false);
  }
  void SetTensorWrapperpath(const paddle::experimental::Tensor& path) {
    path_ = egr::TensorWrapper(path, false);
  }
  void SetTensorWrappercode(const paddle::experimental::Tensor& code) {
    code_ = egr::TensorWrapper(code, false);
  }
  void SetTensorWrapperbias(const paddle::experimental::Tensor& bias) {
    bias_ = egr::TensorWrapper(bias, false);
  }
  void SetTensorWrapperpre_out(const paddle::experimental::Tensor& pre_out) {
    pre_out_ = egr::TensorWrapper(pre_out, false);
  }

  // SetAttributes
  void SetAttributenum_classes(const int& num_classes) {
    num_classes_ = num_classes;
  }
  void SetAttributeremote_prefetch(const bool& remote_prefetch) {
    remote_prefetch_ = remote_prefetch;
  }
  void SetAttributeis_sparse(const bool& is_sparse) { is_sparse_ = is_sparse; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper w_;
  egr::TensorWrapper label_;
  egr::TensorWrapper path_;
  egr::TensorWrapper code_;
  egr::TensorWrapper bias_;
  egr::TensorWrapper pre_out_;

  // Attributes
  int num_classes_;
  bool remote_prefetch_;
  bool is_sparse_;
};

class HuberLossGradNode : public egr::GradNodeBase {
 public:
  HuberLossGradNode() : egr::GradNodeBase() {}
  HuberLossGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~HuberLossGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "HuberLossGradNode"; }

  void ClearTensorWrappers() override {
    residual_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<HuberLossGradNode>(new HuberLossGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperresidual(const paddle::experimental::Tensor& residual) {
    residual_ = egr::TensorWrapper(residual, false);
  }

  // SetAttributes
  void SetAttributedelta(const float& delta) { delta_ = delta; }

 private:
  // TensorWrappers
  egr::TensorWrapper residual_;

  // Attributes
  float delta_;
};

class IndexAddGradNode : public egr::GradNodeBase {
 public:
  IndexAddGradNode() : egr::GradNodeBase() {}
  IndexAddGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~IndexAddGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "IndexAddGradNode"; }

  void ClearTensorWrappers() override {
    index_.clear();
    add_value_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<IndexAddGradNode>(new IndexAddGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperindex(const paddle::experimental::Tensor& index) {
    index_ = egr::TensorWrapper(index, false);
  }
  void SetTensorWrapperadd_value(
      const paddle::experimental::Tensor& add_value) {
    add_value_ = egr::TensorWrapper(add_value, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper index_;
  egr::TensorWrapper add_value_;

  // Attributes
  int axis_;
};

class InstanceNormGradNode : public egr::GradNodeBase {
 public:
  InstanceNormGradNode() : egr::GradNodeBase() {}
  InstanceNormGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~InstanceNormGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "InstanceNormGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    scale_.clear();
    saved_mean_.clear();
    saved_variance_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<InstanceNormGradNode>(new InstanceNormGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperscale(const paddle::experimental::Tensor& scale) {
    scale_ = egr::TensorWrapper(scale, false);
  }
  void SetTensorWrappersaved_mean(
      const paddle::experimental::Tensor& saved_mean) {
    saved_mean_ = egr::TensorWrapper(saved_mean, false);
  }
  void SetTensorWrappersaved_variance(
      const paddle::experimental::Tensor& saved_variance) {
    saved_variance_ = egr::TensorWrapper(saved_variance, false);
  }

  // SetAttributes
  void SetAttributeepsilon(const float& epsilon) { epsilon_ = epsilon; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper scale_;
  egr::TensorWrapper saved_mean_;
  egr::TensorWrapper saved_variance_;

  // Attributes
  float epsilon_;
};

class InstanceNormDoubleGradNode : public egr::GradNodeBase {
 public:
  InstanceNormDoubleGradNode() : egr::GradNodeBase() {}
  InstanceNormDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~InstanceNormDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "InstanceNormDoubleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    fwd_scale_.clear();
    saved_mean_.clear();
    saved_variance_.clear();
    grad_y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<InstanceNormDoubleGradNode>(
        new InstanceNormDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperfwd_scale(
      const paddle::experimental::Tensor& fwd_scale) {
    fwd_scale_ = egr::TensorWrapper(fwd_scale, false);
  }
  void SetTensorWrappersaved_mean(
      const paddle::experimental::Tensor& saved_mean) {
    saved_mean_ = egr::TensorWrapper(saved_mean, false);
  }
  void SetTensorWrappersaved_variance(
      const paddle::experimental::Tensor& saved_variance) {
    saved_variance_ = egr::TensorWrapper(saved_variance, false);
  }
  void SetTensorWrappergrad_y(const paddle::experimental::Tensor& grad_y) {
    grad_y_ = egr::TensorWrapper(grad_y, false);
  }

  // SetAttributes
  void SetAttributeepsilon(const float& epsilon) { epsilon_ = epsilon; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper fwd_scale_;
  egr::TensorWrapper saved_mean_;
  egr::TensorWrapper saved_variance_;
  egr::TensorWrapper grad_y_;

  // Attributes
  float epsilon_;
};

class KldivLossGradNode : public egr::GradNodeBase {
 public:
  KldivLossGradNode() : egr::GradNodeBase() {}
  KldivLossGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~KldivLossGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "KldivLossGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    label_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<KldivLossGradNode>(new KldivLossGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }
  void SetTensorWrapperlabel(const paddle::experimental::Tensor& label) {
    label_ = egr::TensorWrapper(label, false);
  }

  // SetAttributes
  void SetAttributereduction(const std::string& reduction) {
    reduction_ = reduction;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper label_;

  // Attributes
  std::string reduction_;
};

class KronGradNode : public egr::GradNodeBase {
 public:
  KronGradNode() : egr::GradNodeBase() {}
  KronGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~KronGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "KronGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<KronGradNode>(new KronGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
};

class LayerNormGradNode : public egr::GradNodeBase {
 public:
  LayerNormGradNode() : egr::GradNodeBase() {}
  LayerNormGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~LayerNormGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "LayerNormGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    scale_.clear();
    bias_.clear();
    mean_.clear();
    variance_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<LayerNormGradNode>(new LayerNormGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperscale(const paddle::experimental::Tensor& scale) {
    scale_ = egr::TensorWrapper(scale, false);
  }
  void SetTensorWrapperbias(const paddle::experimental::Tensor& bias) {
    bias_ = egr::TensorWrapper(bias, true);
  }
  void SetTensorWrappermean(const paddle::experimental::Tensor& mean) {
    mean_ = egr::TensorWrapper(mean, false);
  }
  void SetTensorWrappervariance(const paddle::experimental::Tensor& variance) {
    variance_ = egr::TensorWrapper(variance, false);
  }

  // SetAttributes
  void SetAttributeepsilon(const float& epsilon) { epsilon_ = epsilon; }
  void SetAttributebegin_norm_axis(const int& begin_norm_axis) {
    begin_norm_axis_ = begin_norm_axis;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper scale_;
  egr::TensorWrapper bias_;
  egr::TensorWrapper mean_;
  egr::TensorWrapper variance_;

  // Attributes
  float epsilon_;
  int begin_norm_axis_;
};

class LinearInterpGradNode : public egr::GradNodeBase {
 public:
  LinearInterpGradNode() : egr::GradNodeBase() {}
  LinearInterpGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~LinearInterpGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "LinearInterpGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_size_.clear();
    for (auto& tw : size_tensor_) {
      tw.clear();
    }
    scale_tensor_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<LinearInterpGradNode>(new LinearInterpGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout_size(const paddle::experimental::Tensor& out_size) {
    out_size_ = egr::TensorWrapper(out_size, false);
  }
  void SetTensorWrappersize_tensor(
      const std::vector<paddle::experimental::Tensor>& size_tensor) {
    for (const auto& eager_tensor : size_tensor) {
      size_tensor_.emplace_back(egr::TensorWrapper(eager_tensor, false));
    };
  }
  void SetTensorWrapperscale_tensor(
      const paddle::experimental::Tensor& scale_tensor) {
    scale_tensor_ = egr::TensorWrapper(scale_tensor, false);
  }

  // SetAttributes
  void SetAttributedata_layout(const std::string& data_layout) {
    data_layout_ = data_layout;
  }
  void SetAttributeout_d(const int& out_d) { out_d_ = out_d; }
  void SetAttributeout_h(const int& out_h) { out_h_ = out_h; }
  void SetAttributeout_w(const int& out_w) { out_w_ = out_w; }
  void SetAttributescale(const std::vector<float>& scale) { scale_ = scale; }
  void SetAttributeinterp_method(const std::string& interp_method) {
    interp_method_ = interp_method;
  }
  void SetAttributealign_corners(const bool& align_corners) {
    align_corners_ = align_corners;
  }
  void SetAttributealign_mode(const int& align_mode) {
    align_mode_ = align_mode;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_size_;
  std::vector<egr::TensorWrapper> size_tensor_;
  egr::TensorWrapper scale_tensor_;

  // Attributes
  std::string data_layout_;
  int out_d_;
  int out_h_;
  int out_w_;
  std::vector<float> scale_;
  std::string interp_method_;
  bool align_corners_;
  int align_mode_;
};

class LogSoftmaxGradNode : public egr::GradNodeBase {
 public:
  LogSoftmaxGradNode() : egr::GradNodeBase() {}
  LogSoftmaxGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~LogSoftmaxGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "LogSoftmaxGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<LogSoftmaxGradNode>(new LogSoftmaxGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
  int axis_;
};

class LogcumsumexpGradNode : public egr::GradNodeBase {
 public:
  LogcumsumexpGradNode() : egr::GradNodeBase() {}
  LogcumsumexpGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~LogcumsumexpGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "LogcumsumexpGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<LogcumsumexpGradNode>(new LogcumsumexpGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }
  void SetAttributeflatten(const bool& flatten) { flatten_ = flatten; }
  void SetAttributeexclusive(const bool& exclusive) { exclusive_ = exclusive; }
  void SetAttributereverse(const bool& reverse) { reverse_ = reverse; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_;

  // Attributes
  int axis_;
  bool flatten_;
  bool exclusive_;
  bool reverse_;
};

class LogsumexpGradNode : public egr::GradNodeBase {
 public:
  LogsumexpGradNode() : egr::GradNodeBase() {}
  LogsumexpGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~LogsumexpGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "LogsumexpGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<LogsumexpGradNode>(new LogsumexpGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributeaxis(const std::vector<int64_t>& axis) { axis_ = axis; }
  void SetAttributekeepdim(const bool& keepdim) { keepdim_ = keepdim; }
  void SetAttributereduce_all(const bool& reduce_all) {
    reduce_all_ = reduce_all;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_;

  // Attributes
  std::vector<int64_t> axis_;
  bool keepdim_;
  bool reduce_all_;
};

class LuGradNode : public egr::GradNodeBase {
 public:
  LuGradNode() : egr::GradNodeBase() {}
  LuGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~LuGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "LuGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_.clear();
    pivots_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<LuGradNode>(new LuGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }
  void SetTensorWrapperpivots(const paddle::experimental::Tensor& pivots) {
    pivots_ = egr::TensorWrapper(pivots, false);
  }

  // SetAttributes
  void SetAttributepivot(const bool& pivot) { pivot_ = pivot; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_;
  egr::TensorWrapper pivots_;

  // Attributes
  bool pivot_;
};

class MarginCrossEntropyGradNode : public egr::GradNodeBase {
 public:
  MarginCrossEntropyGradNode() : egr::GradNodeBase() {}
  MarginCrossEntropyGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MarginCrossEntropyGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MarginCrossEntropyGradNode"; }

  void ClearTensorWrappers() override {
    logits_.clear();
    label_.clear();
    softmax_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<MarginCrossEntropyGradNode>(
        new MarginCrossEntropyGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperlogits(const paddle::experimental::Tensor& logits) {
    logits_ = egr::TensorWrapper(logits, false);
  }
  void SetTensorWrapperlabel(const paddle::experimental::Tensor& label) {
    label_ = egr::TensorWrapper(label, false);
  }
  void SetTensorWrappersoftmax(const paddle::experimental::Tensor& softmax) {
    softmax_ = egr::TensorWrapper(softmax, false);
  }

  // SetAttributes
  void SetAttributereturn_softmax(const bool& return_softmax) {
    return_softmax_ = return_softmax;
  }
  void SetAttributering_id(const int& ring_id) { ring_id_ = ring_id; }
  void SetAttributerank(const int& rank) { rank_ = rank; }
  void SetAttributenranks(const int& nranks) { nranks_ = nranks; }
  void SetAttributemargin1(const float& margin1) { margin1_ = margin1; }
  void SetAttributemargin2(const float& margin2) { margin2_ = margin2; }
  void SetAttributemargin3(const float& margin3) { margin3_ = margin3; }
  void SetAttributescale(const float& scale) { scale_ = scale; }

 private:
  // TensorWrappers
  egr::TensorWrapper logits_;
  egr::TensorWrapper label_;
  egr::TensorWrapper softmax_;

  // Attributes
  bool return_softmax_;
  int ring_id_;
  int rank_;
  int nranks_;
  float margin1_;
  float margin2_;
  float margin3_;
  float scale_;
};

class MatmulGradNode : public egr::GradNodeBase {
 public:
  MatmulGradNode() : egr::GradNodeBase() {}
  MatmulGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MatmulGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MatmulGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MatmulGradNode>(new MatmulGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes
  void SetAttributetranspose_x(const bool& transpose_x) {
    transpose_x_ = transpose_x;
  }
  void SetAttributetranspose_y(const bool& transpose_y) {
    transpose_y_ = transpose_y;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
  bool transpose_x_ = false;
  bool transpose_y_ = false;
};

class MatmulDoubleGradNode : public egr::GradNodeBase {
 public:
  MatmulDoubleGradNode() : egr::GradNodeBase() {}
  MatmulDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MatmulDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MatmulDoubleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MatmulDoubleGradNode>(new MatmulDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }
  void SetTensorWrappergrad_out(const paddle::experimental::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }

  // SetAttributes
  void SetAttributetranspose_x(const bool& transpose_x) {
    transpose_x_ = transpose_x;
  }
  void SetAttributetranspose_y(const bool& transpose_y) {
    transpose_y_ = transpose_y;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;
  egr::TensorWrapper grad_out_;

  // Attributes
  bool transpose_x_ = false;
  bool transpose_y_ = false;
};

class MatmulTripleGradNode : public egr::GradNodeBase {
 public:
  MatmulTripleGradNode() : egr::GradNodeBase() {}
  MatmulTripleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MatmulTripleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MatmulTripleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();
    fwd_grad_out_.clear();
    fwd_grad_grad_x_.clear();
    fwd_grad_grad_y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MatmulTripleGradNode>(new MatmulTripleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }
  void SetTensorWrapperfwd_grad_out(
      const paddle::experimental::Tensor& fwd_grad_out) {
    fwd_grad_out_ = egr::TensorWrapper(fwd_grad_out, false);
  }
  void SetTensorWrapperfwd_grad_grad_x(
      const paddle::experimental::Tensor& fwd_grad_grad_x) {
    fwd_grad_grad_x_ = egr::TensorWrapper(fwd_grad_grad_x, false);
  }
  void SetTensorWrapperfwd_grad_grad_y(
      const paddle::experimental::Tensor& fwd_grad_grad_y) {
    fwd_grad_grad_y_ = egr::TensorWrapper(fwd_grad_grad_y, false);
  }

  // SetAttributes
  void SetAttributetranspose_x(const bool& transpose_x) {
    transpose_x_ = transpose_x;
  }
  void SetAttributetranspose_y(const bool& transpose_y) {
    transpose_y_ = transpose_y;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;
  egr::TensorWrapper fwd_grad_out_;
  egr::TensorWrapper fwd_grad_grad_x_;
  egr::TensorWrapper fwd_grad_grad_y_;

  // Attributes
  bool transpose_x_ = false;
  bool transpose_y_ = false;
};

class MaxGradNode : public egr::GradNodeBase {
 public:
  MaxGradNode() : egr::GradNodeBase() {}
  MaxGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MaxGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MaxGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<MaxGradNode>(new MaxGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributeaxis(const paddle::experimental::IntArray& axis) {
    axis_ = axis;
  }
  void SetAttributekeepdim(const bool& keepdim) { keepdim_ = keepdim; }
  void SetAttributereduce_all(const bool& reduce_all) {
    reduce_all_ = reduce_all;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_;

  // Attributes
  paddle::experimental::IntArray axis_ = {};
  bool keepdim_ = false;
  bool reduce_all_ = false;
};

class MaxPool2dWithIndexGradNode : public egr::GradNodeBase {
 public:
  MaxPool2dWithIndexGradNode() : egr::GradNodeBase() {}
  MaxPool2dWithIndexGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MaxPool2dWithIndexGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MaxPool2dWithIndexGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    mask_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<MaxPool2dWithIndexGradNode>(
        new MaxPool2dWithIndexGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappermask(const paddle::experimental::Tensor& mask) {
    mask_ = egr::TensorWrapper(mask, false);
  }

  // SetAttributes
  void SetAttributekernel_size(const std::vector<int>& kernel_size) {
    kernel_size_ = kernel_size;
  }
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributeglobal_pooling(const bool& global_pooling) {
    global_pooling_ = global_pooling;
  }
  void SetAttributeadaptive(const bool& adaptive) { adaptive_ = adaptive; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper mask_;

  // Attributes
  std::vector<int> kernel_size_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  bool global_pooling_;
  bool adaptive_;
};

class MaxPool3dWithIndexGradNode : public egr::GradNodeBase {
 public:
  MaxPool3dWithIndexGradNode() : egr::GradNodeBase() {}
  MaxPool3dWithIndexGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MaxPool3dWithIndexGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MaxPool3dWithIndexGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    mask_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<MaxPool3dWithIndexGradNode>(
        new MaxPool3dWithIndexGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappermask(const paddle::experimental::Tensor& mask) {
    mask_ = egr::TensorWrapper(mask, false);
  }

  // SetAttributes
  void SetAttributekernel_size(const std::vector<int>& kernel_size) {
    kernel_size_ = kernel_size;
  }
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributeglobal_pooling(const bool& global_pooling) {
    global_pooling_ = global_pooling;
  }
  void SetAttributeadaptive(const bool& adaptive) { adaptive_ = adaptive; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper mask_;

  // Attributes
  std::vector<int> kernel_size_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  bool global_pooling_;
  bool adaptive_;
};

class MaximumGradNode : public egr::GradNodeBase {
 public:
  MaximumGradNode() : egr::GradNodeBase() {}
  MaximumGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MaximumGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MaximumGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MaximumGradNode>(new MaximumGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
  int axis_ = -1;
};

class MeanGradNode : public egr::GradNodeBase {
 public:
  MeanGradNode() : egr::GradNodeBase() {}
  MeanGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MeanGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MeanGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<MeanGradNode>(new MeanGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }

  // SetAttributes
  void SetAttributeaxis(const paddle::experimental::IntArray& axis) {
    axis_ = axis;
  }
  void SetAttributekeepdim(const bool& keepdim) { keepdim_ = keepdim; }
  void SetAttributereduce_all(const bool& reduce_all) {
    reduce_all_ = reduce_all;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  paddle::experimental::IntArray axis_ = {};
  bool keepdim_ = false;
  bool reduce_all_ = false;
};

class MeanDoubleGradNode : public egr::GradNodeBase {
 public:
  MeanDoubleGradNode() : egr::GradNodeBase() {}
  MeanDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MeanDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MeanDoubleGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MeanDoubleGradNode>(new MeanDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeaxis(const paddle::experimental::IntArray& axis) {
    axis_ = axis;
  }
  void SetAttributekeepdim(const bool& keepdim) { keepdim_ = keepdim; }

 private:
  // TensorWrappers

  // Attributes
  paddle::experimental::IntArray axis_ = {};
  bool keepdim_ = false;
};

class MeanAllGradNode : public egr::GradNodeBase {
 public:
  MeanAllGradNode() : egr::GradNodeBase() {}
  MeanAllGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MeanAllGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MeanAllGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MeanAllGradNode>(new MeanAllGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class MeshgridGradNode : public egr::GradNodeBase {
 public:
  MeshgridGradNode() : egr::GradNodeBase() {}
  MeshgridGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MeshgridGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MeshgridGradNode"; }

  void ClearTensorWrappers() override {
    for (auto& tw : inputs_) {
      tw.clear();
    }

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MeshgridGradNode>(new MeshgridGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperinputs(
      const std::vector<paddle::experimental::Tensor>& inputs) {
    for (const auto& eager_tensor : inputs) {
      inputs_.emplace_back(egr::TensorWrapper(eager_tensor, false));
    };
  }

  // SetAttributes

 private:
  // TensorWrappers
  std::vector<egr::TensorWrapper> inputs_;

  // Attributes
};

class MinGradNode : public egr::GradNodeBase {
 public:
  MinGradNode() : egr::GradNodeBase() {}
  MinGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MinGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MinGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<MinGradNode>(new MinGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributeaxis(const paddle::experimental::IntArray& axis) {
    axis_ = axis;
  }
  void SetAttributekeepdim(const bool& keepdim) { keepdim_ = keepdim; }
  void SetAttributereduce_all(const bool& reduce_all) {
    reduce_all_ = reduce_all;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_;

  // Attributes
  paddle::experimental::IntArray axis_ = {};
  bool keepdim_ = false;
  bool reduce_all_ = false;
};

class MinimumGradNode : public egr::GradNodeBase {
 public:
  MinimumGradNode() : egr::GradNodeBase() {}
  MinimumGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MinimumGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MinimumGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MinimumGradNode>(new MinimumGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
  int axis_ = -1;
};

class MishGradNode : public egr::GradNodeBase {
 public:
  MishGradNode() : egr::GradNodeBase() {}
  MishGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MishGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MishGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<MishGradNode>(new MishGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributethreshold(const float& threshold) { threshold_ = threshold; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  float threshold_;
};

class MultiDotGradNode : public egr::GradNodeBase {
 public:
  MultiDotGradNode() : egr::GradNodeBase() {}
  MultiDotGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MultiDotGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MultiDotGradNode"; }

  void ClearTensorWrappers() override {
    for (auto& tw : x_) {
      tw.clear();
    }

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MultiDotGradNode>(new MultiDotGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const std::vector<paddle::experimental::Tensor>& x) {
    for (const auto& eager_tensor : x) {
      x_.emplace_back(egr::TensorWrapper(eager_tensor, false));
    };
  }

  // SetAttributes

 private:
  // TensorWrappers
  std::vector<egr::TensorWrapper> x_;

  // Attributes
};

class MultiplexGradNode : public egr::GradNodeBase {
 public:
  MultiplexGradNode() : egr::GradNodeBase() {}
  MultiplexGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MultiplexGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MultiplexGradNode"; }

  void ClearTensorWrappers() override {
    for (auto& tw : inputs_) {
      tw.clear();
    }
    index_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MultiplexGradNode>(new MultiplexGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperinputs(
      const std::vector<paddle::experimental::Tensor>& inputs) {
    for (const auto& eager_tensor : inputs) {
      inputs_.emplace_back(egr::TensorWrapper(eager_tensor, false));
    };
  }
  void SetTensorWrapperindex(const paddle::experimental::Tensor& index) {
    index_ = egr::TensorWrapper(index, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  std::vector<egr::TensorWrapper> inputs_;
  egr::TensorWrapper index_;

  // Attributes
};

class MultiplyGradNode : public egr::GradNodeBase {
 public:
  MultiplyGradNode() : egr::GradNodeBase() {}
  MultiplyGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MultiplyGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MultiplyGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MultiplyGradNode>(new MultiplyGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
  int axis_ = -1;
};

class MultiplyDoubleGradNode : public egr::GradNodeBase {
 public:
  MultiplyDoubleGradNode() : egr::GradNodeBase() {}
  MultiplyDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MultiplyDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MultiplyDoubleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<MultiplyDoubleGradNode>(
        new MultiplyDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }
  void SetTensorWrappergrad_out(const paddle::experimental::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;
  egr::TensorWrapper grad_out_;

  // Attributes
  int axis_ = -1;
};

class MultiplyTripleGradNode : public egr::GradNodeBase {
 public:
  MultiplyTripleGradNode() : egr::GradNodeBase() {}
  MultiplyTripleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MultiplyTripleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MultiplyTripleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();
    fwd_grad_out_.clear();
    fwd_grad_grad_x_.clear();
    fwd_grad_grad_y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<MultiplyTripleGradNode>(
        new MultiplyTripleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }
  void SetTensorWrapperfwd_grad_out(
      const paddle::experimental::Tensor& fwd_grad_out) {
    fwd_grad_out_ = egr::TensorWrapper(fwd_grad_out, false);
  }
  void SetTensorWrapperfwd_grad_grad_x(
      const paddle::experimental::Tensor& fwd_grad_grad_x) {
    fwd_grad_grad_x_ = egr::TensorWrapper(fwd_grad_grad_x, false);
  }
  void SetTensorWrapperfwd_grad_grad_y(
      const paddle::experimental::Tensor& fwd_grad_grad_y) {
    fwd_grad_grad_y_ = egr::TensorWrapper(fwd_grad_grad_y, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;
  egr::TensorWrapper fwd_grad_out_;
  egr::TensorWrapper fwd_grad_grad_x_;
  egr::TensorWrapper fwd_grad_grad_y_;

  // Attributes
  int axis_ = -1;
};

class NearestInterpGradNode : public egr::GradNodeBase {
 public:
  NearestInterpGradNode() : egr::GradNodeBase() {}
  NearestInterpGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~NearestInterpGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "NearestInterpGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_size_.clear();
    for (auto& tw : size_tensor_) {
      tw.clear();
    }
    scale_tensor_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<NearestInterpGradNode>(
        new NearestInterpGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout_size(const paddle::experimental::Tensor& out_size) {
    out_size_ = egr::TensorWrapper(out_size, false);
  }
  void SetTensorWrappersize_tensor(
      const std::vector<paddle::experimental::Tensor>& size_tensor) {
    for (const auto& eager_tensor : size_tensor) {
      size_tensor_.emplace_back(egr::TensorWrapper(eager_tensor, false));
    };
  }
  void SetTensorWrapperscale_tensor(
      const paddle::experimental::Tensor& scale_tensor) {
    scale_tensor_ = egr::TensorWrapper(scale_tensor, false);
  }

  // SetAttributes
  void SetAttributedata_layout(const std::string& data_layout) {
    data_layout_ = data_layout;
  }
  void SetAttributeout_d(const int& out_d) { out_d_ = out_d; }
  void SetAttributeout_h(const int& out_h) { out_h_ = out_h; }
  void SetAttributeout_w(const int& out_w) { out_w_ = out_w; }
  void SetAttributescale(const std::vector<float>& scale) { scale_ = scale; }
  void SetAttributeinterp_method(const std::string& interp_method) {
    interp_method_ = interp_method;
  }
  void SetAttributealign_corners(const bool& align_corners) {
    align_corners_ = align_corners;
  }
  void SetAttributealign_mode(const int& align_mode) {
    align_mode_ = align_mode;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_size_;
  std::vector<egr::TensorWrapper> size_tensor_;
  egr::TensorWrapper scale_tensor_;

  // Attributes
  std::string data_layout_;
  int out_d_;
  int out_h_;
  int out_w_;
  std::vector<float> scale_;
  std::string interp_method_;
  bool align_corners_;
  int align_mode_;
};

class NormGradNode : public egr::GradNodeBase {
 public:
  NormGradNode() : egr::GradNodeBase() {}
  NormGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~NormGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "NormGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    norm_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<NormGradNode>(new NormGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappernorm(const paddle::experimental::Tensor& norm) {
    norm_ = egr::TensorWrapper(norm, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }
  void SetAttributeepsilon(const float& epsilon) { epsilon_ = epsilon; }
  void SetAttributeis_test(const bool& is_test) { is_test_ = is_test; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper norm_;

  // Attributes
  int axis_;
  float epsilon_;
  bool is_test_;
};

class PNormGradNode : public egr::GradNodeBase {
 public:
  PNormGradNode() : egr::GradNodeBase() {}
  PNormGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~PNormGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "PNormGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<PNormGradNode>(new PNormGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributeporder(const float& porder) { porder_ = porder; }
  void SetAttributeaxis(const int& axis) { axis_ = axis; }
  void SetAttributeepsilon(const float& epsilon) { epsilon_ = epsilon; }
  void SetAttributekeepdim(const bool& keepdim) { keepdim_ = keepdim; }
  void SetAttributeasvector(const bool& asvector) { asvector_ = asvector; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_;

  // Attributes
  float porder_;
  int axis_;
  float epsilon_;
  bool keepdim_;
  bool asvector_;
};

class PadGradNode : public egr::GradNodeBase {
 public:
  PadGradNode() : egr::GradNodeBase() {}
  PadGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~PadGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "PadGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<PadGradNode>(new PadGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }

  // SetAttributes
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributepad_value(const paddle::experimental::Scalar& pad_value) {
    pad_value_ = pad_value;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  std::vector<int> paddings_;
  paddle::experimental::Scalar pad_value_;
};

class PadDoubleGradNode : public egr::GradNodeBase {
 public:
  PadDoubleGradNode() : egr::GradNodeBase() {}
  PadDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~PadDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "PadDoubleGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<PadDoubleGradNode>(new PadDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributepad_value(const paddle::experimental::Scalar& pad_value) {
    pad_value_ = pad_value;
  }

 private:
  // TensorWrappers

  // Attributes
  std::vector<int> paddings_;
  paddle::experimental::Scalar pad_value_;
};

class Pad3dGradNode : public egr::GradNodeBase {
 public:
  Pad3dGradNode() : egr::GradNodeBase() {}
  Pad3dGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Pad3dGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Pad3dGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<Pad3dGradNode>(new Pad3dGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }

  // SetAttributes
  void SetAttributepaddings(const paddle::experimental::IntArray& paddings) {
    paddings_ = paddings;
  }
  void SetAttributemode(const std::string& mode) { mode_ = mode; }
  void SetAttributepad_value(const float& pad_value) { pad_value_ = pad_value; }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  paddle::experimental::IntArray paddings_;
  std::string mode_;
  float pad_value_;
  std::string data_format_;
};

class Pad3dDoubleGradNode : public egr::GradNodeBase {
 public:
  Pad3dDoubleGradNode() : egr::GradNodeBase() {}
  Pad3dDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Pad3dDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Pad3dDoubleGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<Pad3dDoubleGradNode>(new Pad3dDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributepaddings(const paddle::experimental::IntArray& paddings) {
    paddings_ = paddings;
  }
  void SetAttributemode(const std::string& mode) { mode_ = mode; }
  void SetAttributepad_value(const float& pad_value) { pad_value_ = pad_value; }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }

 private:
  // TensorWrappers

  // Attributes
  paddle::experimental::IntArray paddings_;
  std::string mode_;
  float pad_value_;
  std::string data_format_;
};

class Pool2dGradNode : public egr::GradNodeBase {
 public:
  Pool2dGradNode() : egr::GradNodeBase() {}
  Pool2dGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Pool2dGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Pool2dGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<Pool2dGradNode>(new Pool2dGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributekernel_size(
      const paddle::experimental::IntArray& kernel_size) {
    kernel_size_ = kernel_size;
  }
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributeceil_mode(const bool& ceil_mode) { ceil_mode_ = ceil_mode; }
  void SetAttributeexclusive(const bool& exclusive) { exclusive_ = exclusive; }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }
  void SetAttributepooling_type(const std::string& pooling_type) {
    pooling_type_ = pooling_type;
  }
  void SetAttributeglobal_pooling(const bool& global_pooling) {
    global_pooling_ = global_pooling;
  }
  void SetAttributeadaptive(const bool& adaptive) { adaptive_ = adaptive; }
  void SetAttributepadding_algorithm(const std::string& padding_algorithm) {
    padding_algorithm_ = padding_algorithm;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_;

  // Attributes
  paddle::experimental::IntArray kernel_size_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  bool ceil_mode_;
  bool exclusive_;
  std::string data_format_;
  std::string pooling_type_;
  bool global_pooling_;
  bool adaptive_;
  std::string padding_algorithm_;
};

class Pool2dDoubleGradNode : public egr::GradNodeBase {
 public:
  Pool2dDoubleGradNode() : egr::GradNodeBase() {}
  Pool2dDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Pool2dDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Pool2dDoubleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<Pool2dDoubleGradNode>(new Pool2dDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }

  // SetAttributes
  void SetAttributekernel_size(
      const paddle::experimental::IntArray& kernel_size) {
    kernel_size_ = kernel_size;
  }
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributeceil_mode(const bool& ceil_mode) { ceil_mode_ = ceil_mode; }
  void SetAttributeexclusive(const bool& exclusive) { exclusive_ = exclusive; }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }
  void SetAttributepooling_type(const std::string& pooling_type) {
    pooling_type_ = pooling_type;
  }
  void SetAttributeglobal_pooling(const bool& global_pooling) {
    global_pooling_ = global_pooling;
  }
  void SetAttributeadaptive(const bool& adaptive) { adaptive_ = adaptive; }
  void SetAttributepadding_algorithm(const std::string& padding_algorithm) {
    padding_algorithm_ = padding_algorithm;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  paddle::experimental::IntArray kernel_size_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  bool ceil_mode_;
  bool exclusive_;
  std::string data_format_;
  std::string pooling_type_;
  bool global_pooling_;
  bool adaptive_;
  std::string padding_algorithm_;
};

class Pool3dGradNode : public egr::GradNodeBase {
 public:
  Pool3dGradNode() : egr::GradNodeBase() {}
  Pool3dGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Pool3dGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Pool3dGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<Pool3dGradNode>(new Pool3dGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributekernel_size(const std::vector<int>& kernel_size) {
    kernel_size_ = kernel_size;
  }
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributeceil_mode(const bool& ceil_mode) { ceil_mode_ = ceil_mode; }
  void SetAttributeexclusive(const bool& exclusive) { exclusive_ = exclusive; }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }
  void SetAttributepooling_type(const std::string& pooling_type) {
    pooling_type_ = pooling_type;
  }
  void SetAttributeglobal_pooling(const bool& global_pooling) {
    global_pooling_ = global_pooling;
  }
  void SetAttributeadaptive(const bool& adaptive) { adaptive_ = adaptive; }
  void SetAttributepadding_algorithm(const std::string& padding_algorithm) {
    padding_algorithm_ = padding_algorithm;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_;

  // Attributes
  std::vector<int> kernel_size_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  bool ceil_mode_;
  bool exclusive_;
  std::string data_format_;
  std::string pooling_type_;
  bool global_pooling_;
  bool adaptive_;
  std::string padding_algorithm_;
};

class PowGradNode : public egr::GradNodeBase {
 public:
  PowGradNode() : egr::GradNodeBase() {}
  PowGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~PowGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "PowGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<PowGradNode>(new PowGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributey(const paddle::experimental::Scalar& y) { y_ = y; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  paddle::experimental::Scalar y_ = -1;
};

class PowDoubleGradNode : public egr::GradNodeBase {
 public:
  PowDoubleGradNode() : egr::GradNodeBase() {}
  PowDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~PowDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "PowDoubleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<PowDoubleGradNode>(new PowDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappergrad_out(const paddle::experimental::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }

  // SetAttributes
  void SetAttributey(const paddle::experimental::Scalar& y) { y_ = y; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper grad_out_;

  // Attributes
  paddle::experimental::Scalar y_;
};

class PowTripleGradNode : public egr::GradNodeBase {
 public:
  PowTripleGradNode() : egr::GradNodeBase() {}
  PowTripleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~PowTripleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "PowTripleGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    grad_out_.clear();
    grad_grad_x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<PowTripleGradNode>(new PowTripleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappergrad_out(const paddle::experimental::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }
  void SetTensorWrappergrad_grad_x(
      const paddle::experimental::Tensor& grad_grad_x) {
    grad_grad_x_ = egr::TensorWrapper(grad_grad_x, false);
  }

  // SetAttributes
  void SetAttributey(const paddle::experimental::Scalar& y) { y_ = y; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper grad_out_;
  egr::TensorWrapper grad_grad_x_;

  // Attributes
  paddle::experimental::Scalar y_;
};

class PreluGradNode : public egr::GradNodeBase {
 public:
  PreluGradNode() : egr::GradNodeBase() {}
  PreluGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~PreluGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "PreluGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    alpha_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<PreluGradNode>(new PreluGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperalpha(const paddle::experimental::Tensor& alpha) {
    alpha_ = egr::TensorWrapper(alpha, false);
  }

  // SetAttributes
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }
  void SetAttributemode(const std::string& mode) { mode_ = mode; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper alpha_;

  // Attributes
  std::string data_format_;
  std::string mode_;
};

class ProdGradNode : public egr::GradNodeBase {
 public:
  ProdGradNode() : egr::GradNodeBase() {}
  ProdGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ProdGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ProdGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<ProdGradNode>(new ProdGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributedims(const paddle::experimental::IntArray& dims) {
    dims_ = dims;
  }
  void SetAttributekeep_dim(const bool& keep_dim) { keep_dim_ = keep_dim; }
  void SetAttributereduce_all(const bool& reduce_all) {
    reduce_all_ = reduce_all;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_;

  // Attributes
  paddle::experimental::IntArray dims_;
  bool keep_dim_;
  bool reduce_all_;
};

class PsroiPoolGradNode : public egr::GradNodeBase {
 public:
  PsroiPoolGradNode() : egr::GradNodeBase() {}
  PsroiPoolGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~PsroiPoolGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "PsroiPoolGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    boxes_.clear();
    boxes_num_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<PsroiPoolGradNode>(new PsroiPoolGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperboxes(const paddle::experimental::Tensor& boxes) {
    boxes_ = egr::TensorWrapper(boxes, false);
  }
  void SetTensorWrapperboxes_num(
      const paddle::experimental::Tensor& boxes_num) {
    boxes_num_ = egr::TensorWrapper(boxes_num, false);
  }

  // SetAttributes
  void SetAttributepooled_height(const int& pooled_height) {
    pooled_height_ = pooled_height;
  }
  void SetAttributepooled_width(const int& pooled_width) {
    pooled_width_ = pooled_width;
  }
  void SetAttributeoutput_channels(const int& output_channels) {
    output_channels_ = output_channels;
  }
  void SetAttributespatial_scale(const float& spatial_scale) {
    spatial_scale_ = spatial_scale;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper boxes_;
  egr::TensorWrapper boxes_num_;

  // Attributes
  int pooled_height_;
  int pooled_width_;
  int output_channels_;
  float spatial_scale_;
};

class Relu6GradNode : public egr::GradNodeBase {
 public:
  Relu6GradNode() : egr::GradNodeBase() {}
  Relu6GradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Relu6GradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Relu6GradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<Relu6GradNode>(new Relu6GradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributethreshold(const float& threshold) { threshold_ = threshold; }

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
  float threshold_ = 6;
};

class RepeatInterleaveGradNode : public egr::GradNodeBase {
 public:
  RepeatInterleaveGradNode() : egr::GradNodeBase() {}
  RepeatInterleaveGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~RepeatInterleaveGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "RepeatInterleaveGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<RepeatInterleaveGradNode>(
        new RepeatInterleaveGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributerepeats(const int& repeats) { repeats_ = repeats; }
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  int repeats_;
  int axis_;
};

class RepeatInterleaveWithTensorIndexGradNode : public egr::GradNodeBase {
 public:
  RepeatInterleaveWithTensorIndexGradNode() : egr::GradNodeBase() {}
  RepeatInterleaveWithTensorIndexGradNode(size_t bwd_in_slot_num,
                                          size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~RepeatInterleaveWithTensorIndexGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override {
    return "RepeatInterleaveWithTensorIndexGradNode";
  }

  void ClearTensorWrappers() override {
    x_.clear();
    repeats_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<RepeatInterleaveWithTensorIndexGradNode>(
        new RepeatInterleaveWithTensorIndexGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperrepeats(const paddle::experimental::Tensor& repeats) {
    repeats_ = egr::TensorWrapper(repeats, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper repeats_;

  // Attributes
  int axis_;
};

class ReshapeGradNode : public egr::GradNodeBase {
 public:
  ReshapeGradNode() : egr::GradNodeBase() {}
  ReshapeGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ReshapeGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ReshapeGradNode"; }

  void ClearTensorWrappers() override {
    xshape_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ReshapeGradNode>(new ReshapeGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperxshape(const paddle::experimental::Tensor& xshape) {
    xshape_ = egr::TensorWrapper(xshape, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper xshape_;

  // Attributes
};

class ReshapeDoubleGradNode : public egr::GradNodeBase {
 public:
  ReshapeDoubleGradNode() : egr::GradNodeBase() {}
  ReshapeDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ReshapeDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ReshapeDoubleGradNode"; }

  void ClearTensorWrappers() override {
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<ReshapeDoubleGradNode>(
        new ReshapeDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrappergrad_out(const paddle::experimental::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, true);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper grad_out_;

  // Attributes
};

class ReverseGradNode : public egr::GradNodeBase {
 public:
  ReverseGradNode() : egr::GradNodeBase() {}
  ReverseGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ReverseGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ReverseGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ReverseGradNode>(new ReverseGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeaxis(const paddle::experimental::IntArray& axis) {
    axis_ = axis;
  }

 private:
  // TensorWrappers

  // Attributes
  paddle::experimental::IntArray axis_;
};

class RnnGradNode : public egr::GradNodeBase {
 public:
  RnnGradNode() : egr::GradNodeBase() {}
  RnnGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~RnnGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "RnnGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    for (auto& tw : pre_state_) {
      tw.clear();
    }
    for (auto& tw : weight_list_) {
      tw.clear();
    }
    sequence_length_.clear();
    out_.clear();
    dropout_state_out_.clear();
    reserve_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<RnnGradNode>(new RnnGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperpre_state(
      const std::vector<paddle::experimental::Tensor>& pre_state) {
    for (const auto& eager_tensor : pre_state) {
      pre_state_.emplace_back(egr::TensorWrapper(eager_tensor, false));
    };
  }
  void SetTensorWrapperweight_list(
      const std::vector<paddle::experimental::Tensor>& weight_list) {
    for (const auto& eager_tensor : weight_list) {
      weight_list_.emplace_back(egr::TensorWrapper(eager_tensor, false));
    };
  }
  void SetTensorWrappersequence_length(
      const paddle::experimental::Tensor& sequence_length) {
    sequence_length_ = egr::TensorWrapper(sequence_length, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }
  void SetTensorWrapperdropout_state_out(
      const paddle::experimental::Tensor& dropout_state_out) {
    dropout_state_out_ = egr::TensorWrapper(dropout_state_out, false);
  }
  void SetTensorWrapperreserve(const paddle::experimental::Tensor& reserve) {
    reserve_ = egr::TensorWrapper(reserve, false);
  }

  // SetAttributes
  void SetAttributedropout_prob(const float& dropout_prob) {
    dropout_prob_ = dropout_prob;
  }
  void SetAttributeis_bidirec(const bool& is_bidirec) {
    is_bidirec_ = is_bidirec;
  }
  void SetAttributeinput_size(const int& input_size) {
    input_size_ = input_size;
  }
  void SetAttributehidden_size(const int& hidden_size) {
    hidden_size_ = hidden_size;
  }
  void SetAttributenum_layers(const int& num_layers) {
    num_layers_ = num_layers;
  }
  void SetAttributemode(const std::string& mode) { mode_ = mode; }
  void SetAttributeseed(const int& seed) { seed_ = seed; }
  void SetAttributeis_test(const bool& is_test) { is_test_ = is_test; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  std::vector<egr::TensorWrapper> pre_state_;
  std::vector<egr::TensorWrapper> weight_list_;
  egr::TensorWrapper sequence_length_;
  egr::TensorWrapper out_;
  egr::TensorWrapper dropout_state_out_;
  egr::TensorWrapper reserve_;

  // Attributes
  float dropout_prob_;
  bool is_bidirec_;
  int input_size_;
  int hidden_size_;
  int num_layers_;
  std::string mode_;
  int seed_;
  bool is_test_;
};

class RoiAlignGradNode : public egr::GradNodeBase {
 public:
  RoiAlignGradNode() : egr::GradNodeBase() {}
  RoiAlignGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~RoiAlignGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "RoiAlignGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    boxes_.clear();
    boxes_num_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<RoiAlignGradNode>(new RoiAlignGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }
  void SetTensorWrapperboxes(const paddle::experimental::Tensor& boxes) {
    boxes_ = egr::TensorWrapper(boxes, false);
  }
  void SetTensorWrapperboxes_num(
      const paddle::experimental::Tensor& boxes_num) {
    boxes_num_ = egr::TensorWrapper(boxes_num, false);
  }

  // SetAttributes
  void SetAttributepooled_height(const int& pooled_height) {
    pooled_height_ = pooled_height;
  }
  void SetAttributepooled_width(const int& pooled_width) {
    pooled_width_ = pooled_width;
  }
  void SetAttributespatial_scale(const float& spatial_scale) {
    spatial_scale_ = spatial_scale;
  }
  void SetAttributesampling_ratio(const int& sampling_ratio) {
    sampling_ratio_ = sampling_ratio;
  }
  void SetAttributealigned(const bool& aligned) { aligned_ = aligned; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper boxes_;
  egr::TensorWrapper boxes_num_;

  // Attributes
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
  int sampling_ratio_;
  bool aligned_;
};

class RoiPoolGradNode : public egr::GradNodeBase {
 public:
  RoiPoolGradNode() : egr::GradNodeBase() {}
  RoiPoolGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~RoiPoolGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "RoiPoolGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    boxes_.clear();
    boxes_num_.clear();
    arg_max_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<RoiPoolGradNode>(new RoiPoolGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperboxes(const paddle::experimental::Tensor& boxes) {
    boxes_ = egr::TensorWrapper(boxes, false);
  }
  void SetTensorWrapperboxes_num(
      const paddle::experimental::Tensor& boxes_num) {
    boxes_num_ = egr::TensorWrapper(boxes_num, false);
  }
  void SetTensorWrapperarg_max(const paddle::experimental::Tensor& arg_max) {
    arg_max_ = egr::TensorWrapper(arg_max, false);
  }

  // SetAttributes
  void SetAttributepooled_height(const int& pooled_height) {
    pooled_height_ = pooled_height;
  }
  void SetAttributepooled_width(const int& pooled_width) {
    pooled_width_ = pooled_width;
  }
  void SetAttributespatial_scale(const float& spatial_scale) {
    spatial_scale_ = spatial_scale;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper boxes_;
  egr::TensorWrapper boxes_num_;
  egr::TensorWrapper arg_max_;

  // Attributes
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

class ScaleGradNode : public egr::GradNodeBase {
 public:
  ScaleGradNode() : egr::GradNodeBase() {}
  ScaleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ScaleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ScaleGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<ScaleGradNode>(new ScaleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributescale(const paddle::experimental::Scalar& scale) {
    scale_ = scale;
  }
  void SetAttributebias_after_scale(const bool& bias_after_scale) {
    bias_after_scale_ = bias_after_scale;
  }

 private:
  // TensorWrappers

  // Attributes
  paddle::experimental::Scalar scale_ = 1.0;
  bool bias_after_scale_ = true;
};

class SegmentPoolGradNode : public egr::GradNodeBase {
 public:
  SegmentPoolGradNode() : egr::GradNodeBase() {}
  SegmentPoolGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SegmentPoolGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SegmentPoolGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    segment_ids_.clear();
    out_.clear();
    summed_ids_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SegmentPoolGradNode>(new SegmentPoolGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappersegment_ids(
      const paddle::experimental::Tensor& segment_ids) {
    segment_ids_ = egr::TensorWrapper(segment_ids, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }
  void SetTensorWrappersummed_ids(
      const paddle::experimental::Tensor& summed_ids) {
    summed_ids_ = egr::TensorWrapper(summed_ids, false);
  }

  // SetAttributes
  void SetAttributepooltype(const std::string& pooltype) {
    pooltype_ = pooltype;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper segment_ids_;
  egr::TensorWrapper out_;
  egr::TensorWrapper summed_ids_;

  // Attributes
  std::string pooltype_;
};

class SendURecvGradNode : public egr::GradNodeBase {
 public:
  SendURecvGradNode() : egr::GradNodeBase() {}
  SendURecvGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SendURecvGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SendURecvGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    src_index_.clear();
    dst_index_.clear();
    out_.clear();
    dst_count_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SendURecvGradNode>(new SendURecvGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappersrc_index(
      const paddle::experimental::Tensor& src_index) {
    src_index_ = egr::TensorWrapper(src_index, false);
  }
  void SetTensorWrapperdst_index(
      const paddle::experimental::Tensor& dst_index) {
    dst_index_ = egr::TensorWrapper(dst_index, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }
  void SetTensorWrapperdst_count(
      const paddle::experimental::Tensor& dst_count) {
    dst_count_ = egr::TensorWrapper(dst_count, false);
  }

  // SetAttributes
  void SetAttributereduce_op(const std::string& reduce_op) {
    reduce_op_ = reduce_op;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper src_index_;
  egr::TensorWrapper dst_index_;
  egr::TensorWrapper out_;
  egr::TensorWrapper dst_count_;

  // Attributes
  std::string reduce_op_ = "SUM";
};

class SendUeRecvGradNode : public egr::GradNodeBase {
 public:
  SendUeRecvGradNode() : egr::GradNodeBase() {}
  SendUeRecvGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SendUeRecvGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SendUeRecvGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();
    src_index_.clear();
    dst_index_.clear();
    out_.clear();
    dst_count_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SendUeRecvGradNode>(new SendUeRecvGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }
  void SetTensorWrappersrc_index(
      const paddle::experimental::Tensor& src_index) {
    src_index_ = egr::TensorWrapper(src_index, false);
  }
  void SetTensorWrapperdst_index(
      const paddle::experimental::Tensor& dst_index) {
    dst_index_ = egr::TensorWrapper(dst_index, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }
  void SetTensorWrapperdst_count(
      const paddle::experimental::Tensor& dst_count) {
    dst_count_ = egr::TensorWrapper(dst_count, false);
  }

  // SetAttributes
  void SetAttributemessage_op(const std::string& message_op) {
    message_op_ = message_op;
  }
  void SetAttributereduce_op(const std::string& reduce_op) {
    reduce_op_ = reduce_op;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;
  egr::TensorWrapper src_index_;
  egr::TensorWrapper dst_index_;
  egr::TensorWrapper out_;
  egr::TensorWrapper dst_count_;

  // Attributes
  std::string message_op_;
  std::string reduce_op_;
};

class SigmoidCrossEntropyWithLogitsGradNode : public egr::GradNodeBase {
 public:
  SigmoidCrossEntropyWithLogitsGradNode() : egr::GradNodeBase() {}
  SigmoidCrossEntropyWithLogitsGradNode(size_t bwd_in_slot_num,
                                        size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SigmoidCrossEntropyWithLogitsGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override {
    return "SigmoidCrossEntropyWithLogitsGradNode";
  }

  void ClearTensorWrappers() override {
    x_.clear();
    label_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SigmoidCrossEntropyWithLogitsGradNode>(
        new SigmoidCrossEntropyWithLogitsGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperlabel(const paddle::experimental::Tensor& label) {
    label_ = egr::TensorWrapper(label, false);
  }

  // SetAttributes
  void SetAttributenormalize(const bool& normalize) { normalize_ = normalize; }
  void SetAttributeignore_index(const int& ignore_index) {
    ignore_index_ = ignore_index;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper label_;

  // Attributes
  bool normalize_;
  int ignore_index_;
};

class SignGradNode : public egr::GradNodeBase {
 public:
  SignGradNode() : egr::GradNodeBase() {}
  SignGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SignGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SignGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SignGradNode>(new SignGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes

 private:
  // TensorWrappers

  // Attributes
};

class SliceGradNode : public egr::GradNodeBase {
 public:
  SliceGradNode() : egr::GradNodeBase() {}
  SliceGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SliceGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SliceGradNode"; }

  void ClearTensorWrappers() override {
    input_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SliceGradNode>(new SliceGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperinput(const paddle::experimental::Tensor& input) {
    input_ = egr::TensorWrapper(input, true);
  }

  // SetAttributes
  void SetAttributeaxes(const std::vector<int64_t>& axes) { axes_ = axes; }
  void SetAttributestarts(const paddle::experimental::IntArray& starts) {
    starts_ = starts;
  }
  void SetAttributeends(const paddle::experimental::IntArray& ends) {
    ends_ = ends;
  }
  void SetAttributeinfer_flags(const std::vector<int64_t>& infer_flags) {
    infer_flags_ = infer_flags;
  }
  void SetAttributedecrease_axis(const std::vector<int64_t>& decrease_axis) {
    decrease_axis_ = decrease_axis;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;

  // Attributes
  std::vector<int64_t> axes_;
  paddle::experimental::IntArray starts_;
  paddle::experimental::IntArray ends_;
  std::vector<int64_t> infer_flags_;
  std::vector<int64_t> decrease_axis_;
};

class SliceDoubleGradNode : public egr::GradNodeBase {
 public:
  SliceDoubleGradNode() : egr::GradNodeBase() {}
  SliceDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SliceDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SliceDoubleGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SliceDoubleGradNode>(new SliceDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeaxes(const std::vector<int64_t>& axes) { axes_ = axes; }
  void SetAttributestarts(const paddle::experimental::IntArray& starts) {
    starts_ = starts;
  }
  void SetAttributeends(const paddle::experimental::IntArray& ends) {
    ends_ = ends;
  }
  void SetAttributeinfer_flags(const std::vector<int64_t>& infer_flags) {
    infer_flags_ = infer_flags;
  }
  void SetAttributedecrease_axis(const std::vector<int64_t>& decrease_axis) {
    decrease_axis_ = decrease_axis;
  }

 private:
  // TensorWrappers

  // Attributes
  std::vector<int64_t> axes_;
  paddle::experimental::IntArray starts_;
  paddle::experimental::IntArray ends_;
  std::vector<int64_t> infer_flags_;
  std::vector<int64_t> decrease_axis_;
};

class SlogdetGradNode : public egr::GradNodeBase {
 public:
  SlogdetGradNode() : egr::GradNodeBase() {}
  SlogdetGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SlogdetGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SlogdetGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SlogdetGradNode>(new SlogdetGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_;

  // Attributes
};

class SoftmaxGradNode : public egr::GradNodeBase {
 public:
  SoftmaxGradNode() : egr::GradNodeBase() {}
  SoftmaxGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SoftmaxGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SoftmaxGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SoftmaxGradNode>(new SoftmaxGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
  int axis_;
};

class SpectralNormGradNode : public egr::GradNodeBase {
 public:
  SpectralNormGradNode() : egr::GradNodeBase() {}
  SpectralNormGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SpectralNormGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SpectralNormGradNode"; }

  void ClearTensorWrappers() override {
    weight_.clear();
    u_.clear();
    v_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SpectralNormGradNode>(new SpectralNormGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperweight(const paddle::experimental::Tensor& weight) {
    weight_ = egr::TensorWrapper(weight, false);
  }
  void SetTensorWrapperu(const paddle::experimental::Tensor& u) {
    u_ = egr::TensorWrapper(u, false);
  }
  void SetTensorWrapperv(const paddle::experimental::Tensor& v) {
    v_ = egr::TensorWrapper(v, false);
  }

  // SetAttributes
  void SetAttributedim(const int& dim) { dim_ = dim; }
  void SetAttributepower_iters(const int& power_iters) {
    power_iters_ = power_iters;
  }
  void SetAttributeeps(const float& eps) { eps_ = eps; }

 private:
  // TensorWrappers
  egr::TensorWrapper weight_;
  egr::TensorWrapper u_;
  egr::TensorWrapper v_;

  // Attributes
  int dim_;
  int power_iters_;
  float eps_;
};

class SplitGradNode : public egr::GradNodeBase {
 public:
  SplitGradNode() : egr::GradNodeBase() {}
  SplitGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SplitGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SplitGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SplitGradNode>(new SplitGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeaxis(const paddle::experimental::Scalar& axis) {
    axis_ = axis;
  }

 private:
  // TensorWrappers

  // Attributes
  paddle::experimental::Scalar axis_ = -1;
};

class SplitWithNumGradNode : public egr::GradNodeBase {
 public:
  SplitWithNumGradNode() : egr::GradNodeBase() {}
  SplitWithNumGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SplitWithNumGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SplitWithNumGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SplitWithNumGradNode>(new SplitWithNumGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeaxis(const paddle::experimental::Scalar& axis) {
    axis_ = axis;
  }

 private:
  // TensorWrappers

  // Attributes
  paddle::experimental::Scalar axis_ = -1;
};

class SquaredL2NormGradNode : public egr::GradNodeBase {
 public:
  SquaredL2NormGradNode() : egr::GradNodeBase() {}
  SquaredL2NormGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SquaredL2NormGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SquaredL2NormGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SquaredL2NormGradNode>(
        new SquaredL2NormGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class StackGradNode : public egr::GradNodeBase {
 public:
  StackGradNode() : egr::GradNodeBase() {}
  StackGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~StackGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "StackGradNode"; }

  void ClearTensorWrappers() override {
    for (auto& tw : x_) {
      tw.clear();
    }

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<StackGradNode>(new StackGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const std::vector<paddle::experimental::Tensor>& x) {
    for (const auto& eager_tensor : x) {
      x_.emplace_back(egr::TensorWrapper(eager_tensor, true));
    };
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  std::vector<egr::TensorWrapper> x_;

  // Attributes
  int axis_;
};

class StridedSliceGradNode : public egr::GradNodeBase {
 public:
  StridedSliceGradNode() : egr::GradNodeBase() {}
  StridedSliceGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~StridedSliceGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "StridedSliceGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<StridedSliceGradNode>(new StridedSliceGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }

  // SetAttributes
  void SetAttributeaxes(const std::vector<int>& axes) { axes_ = axes; }
  void SetAttributestarts(const paddle::experimental::IntArray& starts) {
    starts_ = starts;
  }
  void SetAttributeends(const paddle::experimental::IntArray& ends) {
    ends_ = ends;
  }
  void SetAttributestrides(const paddle::experimental::IntArray& strides) {
    strides_ = strides;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  std::vector<int> axes_;
  paddle::experimental::IntArray starts_;
  paddle::experimental::IntArray ends_;
  paddle::experimental::IntArray strides_;
};

class SubtractGradNode : public egr::GradNodeBase {
 public:
  SubtractGradNode() : egr::GradNodeBase() {}
  SubtractGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SubtractGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SubtractGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SubtractGradNode>(new SubtractGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, true);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
  int axis_ = -1;
};

class SubtractDoubleGradNode : public egr::GradNodeBase {
 public:
  SubtractDoubleGradNode() : egr::GradNodeBase() {}
  SubtractDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SubtractDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SubtractDoubleGradNode"; }

  void ClearTensorWrappers() override {
    y_.clear();
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SubtractDoubleGradNode>(
        new SubtractDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, true);
  }
  void SetTensorWrappergrad_out(const paddle::experimental::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, true);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper y_;
  egr::TensorWrapper grad_out_;

  // Attributes
  int axis_ = -1;
};

class SumGradNode : public egr::GradNodeBase {
 public:
  SumGradNode() : egr::GradNodeBase() {}
  SumGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SumGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SumGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SumGradNode>(new SumGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }

  // SetAttributes
  void SetAttributeaxis(const paddle::experimental::IntArray& axis) {
    axis_ = axis;
  }
  void SetAttributekeepdim(const bool& keepdim) { keepdim_ = keepdim; }
  void SetAttributereduce_all(const bool& reduce_all) {
    reduce_all_ = reduce_all;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  paddle::experimental::IntArray axis_;
  bool keepdim_;
  bool reduce_all_ = false;
};

class SumDoubleGradNode : public egr::GradNodeBase {
 public:
  SumDoubleGradNode() : egr::GradNodeBase() {}
  SumDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SumDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SumDoubleGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SumDoubleGradNode>(new SumDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeaxis(const paddle::experimental::IntArray& axis) {
    axis_ = axis;
  }
  void SetAttributekeepdim(const bool& keepdim) { keepdim_ = keepdim; }

 private:
  // TensorWrappers

  // Attributes
  paddle::experimental::IntArray axis_ = {};
  bool keepdim_ = false;
};

class SwishGradNode : public egr::GradNodeBase {
 public:
  SwishGradNode() : egr::GradNodeBase() {}
  SwishGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SwishGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SwishGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SwishGradNode>(new SwishGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributebete(const float& bete) { bete_ = bete; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  float bete_ = 1.0;
};

class SyncBatchNormGradNode : public egr::GradNodeBase {
 public:
  SyncBatchNormGradNode() : egr::GradNodeBase() {}
  SyncBatchNormGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SyncBatchNormGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SyncBatchNormGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    scale_.clear();
    bias_.clear();
    saved_mean_.clear();
    saved_variance_.clear();
    reserve_space_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SyncBatchNormGradNode>(
        new SyncBatchNormGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperscale(const paddle::experimental::Tensor& scale) {
    scale_ = egr::TensorWrapper(scale, false);
  }
  void SetTensorWrapperbias(const paddle::experimental::Tensor& bias) {
    bias_ = egr::TensorWrapper(bias, false);
  }
  void SetTensorWrappersaved_mean(
      const paddle::experimental::Tensor& saved_mean) {
    saved_mean_ = egr::TensorWrapper(saved_mean, false);
  }
  void SetTensorWrappersaved_variance(
      const paddle::experimental::Tensor& saved_variance) {
    saved_variance_ = egr::TensorWrapper(saved_variance, false);
  }
  void SetTensorWrapperreserve_space(
      const paddle::experimental::Tensor& reserve_space) {
    reserve_space_ = egr::TensorWrapper(reserve_space, false);
  }

  // SetAttributes
  void SetAttributemomentum(const float& momentum) { momentum_ = momentum; }
  void SetAttributeepsilon(const float& epsilon) { epsilon_ = epsilon; }
  void SetAttributedata_layout(const std::string& data_layout) {
    data_layout_ = data_layout;
  }
  void SetAttributeis_test(const bool& is_test) { is_test_ = is_test; }
  void SetAttributeuse_global_stats(const bool& use_global_stats) {
    use_global_stats_ = use_global_stats;
  }
  void SetAttributetrainable_statistics(const bool& trainable_statistics) {
    trainable_statistics_ = trainable_statistics;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper scale_;
  egr::TensorWrapper bias_;
  egr::TensorWrapper saved_mean_;
  egr::TensorWrapper saved_variance_;
  egr::TensorWrapper reserve_space_;

  // Attributes
  float momentum_;
  float epsilon_;
  std::string data_layout_;
  bool is_test_;
  bool use_global_stats_;
  bool trainable_statistics_;
};

class TemporalShiftGradNode : public egr::GradNodeBase {
 public:
  TemporalShiftGradNode() : egr::GradNodeBase() {}
  TemporalShiftGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TemporalShiftGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TemporalShiftGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<TemporalShiftGradNode>(
        new TemporalShiftGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeseg_num(const int& seg_num) { seg_num_ = seg_num; }
  void SetAttributeshift_ratio(const float& shift_ratio) {
    shift_ratio_ = shift_ratio;
  }
  void SetAttributedata_format_str(const std::string& data_format_str) {
    data_format_str_ = data_format_str;
  }

 private:
  // TensorWrappers

  // Attributes
  int seg_num_;
  float shift_ratio_;
  std::string data_format_str_;
};

class TileGradNode : public egr::GradNodeBase {
 public:
  TileGradNode() : egr::GradNodeBase() {}
  TileGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TileGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TileGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<TileGradNode>(new TileGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, true);
  }

  // SetAttributes
  void SetAttributerepeat_times(
      const paddle::experimental::IntArray& repeat_times) {
    repeat_times_ = repeat_times;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  paddle::experimental::IntArray repeat_times_;
};

class TileDoubleGradNode : public egr::GradNodeBase {
 public:
  TileDoubleGradNode() : egr::GradNodeBase() {}
  TileDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TileDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TileDoubleGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<TileDoubleGradNode>(new TileDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributerepeat_times(
      const paddle::experimental::IntArray& repeat_times) {
    repeat_times_ = repeat_times;
  }

 private:
  // TensorWrappers

  // Attributes
  paddle::experimental::IntArray repeat_times_;
};

class TransposeGradNode : public egr::GradNodeBase {
 public:
  TransposeGradNode() : egr::GradNodeBase() {}
  TransposeGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TransposeGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TransposeGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<TransposeGradNode>(new TransposeGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeperm(const std::vector<int>& perm) { perm_ = perm; }

 private:
  // TensorWrappers

  // Attributes
  std::vector<int> perm_;
};

class TransposeDoubleGradNode : public egr::GradNodeBase {
 public:
  TransposeDoubleGradNode() : egr::GradNodeBase() {}
  TransposeDoubleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TransposeDoubleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TransposeDoubleGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<TransposeDoubleGradNode>(
        new TransposeDoubleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeperm(const std::vector<int>& perm) { perm_ = perm; }

 private:
  // TensorWrappers

  // Attributes
  std::vector<int> perm_;
};

class TriangularSolveGradNode : public egr::GradNodeBase {
 public:
  TriangularSolveGradNode() : egr::GradNodeBase() {}
  TriangularSolveGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TriangularSolveGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TriangularSolveGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<TriangularSolveGradNode>(
        new TriangularSolveGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributeupper(const bool& upper) { upper_ = upper; }
  void SetAttributetranpose(const bool& tranpose) { tranpose_ = tranpose; }
  void SetAttributeunitriangular(const bool& unitriangular) {
    unitriangular_ = unitriangular;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;
  egr::TensorWrapper out_;

  // Attributes
  bool upper_;
  bool tranpose_;
  bool unitriangular_;
};

class TrilGradNode : public egr::GradNodeBase {
 public:
  TrilGradNode() : egr::GradNodeBase() {}
  TrilGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TrilGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TrilGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<TrilGradNode>(new TrilGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributediagonal(const int& diagonal) { diagonal_ = diagonal; }

 private:
  // TensorWrappers

  // Attributes
  int diagonal_;
};

class TrilinearInterpGradNode : public egr::GradNodeBase {
 public:
  TrilinearInterpGradNode() : egr::GradNodeBase() {}
  TrilinearInterpGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TrilinearInterpGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TrilinearInterpGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    out_size_.clear();
    for (auto& tw : size_tensor_) {
      tw.clear();
    }
    scale_tensor_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<TrilinearInterpGradNode>(
        new TrilinearInterpGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperout_size(const paddle::experimental::Tensor& out_size) {
    out_size_ = egr::TensorWrapper(out_size, false);
  }
  void SetTensorWrappersize_tensor(
      const std::vector<paddle::experimental::Tensor>& size_tensor) {
    for (const auto& eager_tensor : size_tensor) {
      size_tensor_.emplace_back(egr::TensorWrapper(eager_tensor, false));
    };
  }
  void SetTensorWrapperscale_tensor(
      const paddle::experimental::Tensor& scale_tensor) {
    scale_tensor_ = egr::TensorWrapper(scale_tensor, false);
  }

  // SetAttributes
  void SetAttributedata_layout(const std::string& data_layout) {
    data_layout_ = data_layout;
  }
  void SetAttributeout_d(const int& out_d) { out_d_ = out_d; }
  void SetAttributeout_h(const int& out_h) { out_h_ = out_h; }
  void SetAttributeout_w(const int& out_w) { out_w_ = out_w; }
  void SetAttributescale(const std::vector<float>& scale) { scale_ = scale; }
  void SetAttributeinterp_method(const std::string& interp_method) {
    interp_method_ = interp_method;
  }
  void SetAttributealign_corners(const bool& align_corners) {
    align_corners_ = align_corners;
  }
  void SetAttributealign_mode(const int& align_mode) {
    align_mode_ = align_mode;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper out_size_;
  std::vector<egr::TensorWrapper> size_tensor_;
  egr::TensorWrapper scale_tensor_;

  // Attributes
  std::string data_layout_;
  int out_d_;
  int out_h_;
  int out_w_;
  std::vector<float> scale_;
  std::string interp_method_;
  bool align_corners_;
  int align_mode_;
};

class TriuGradNode : public egr::GradNodeBase {
 public:
  TriuGradNode() : egr::GradNodeBase() {}
  TriuGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TriuGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TriuGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<TriuGradNode>(new TriuGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributediagonal(const int& diagonal) { diagonal_ = diagonal; }

 private:
  // TensorWrappers

  // Attributes
  int diagonal_;
};

class UnbindGradNode : public egr::GradNodeBase {
 public:
  UnbindGradNode() : egr::GradNodeBase() {}
  UnbindGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~UnbindGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "UnbindGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<UnbindGradNode>(new UnbindGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers

  // Attributes
  int axis_;
};

class UniformInplaceGradNode : public egr::GradNodeBase {
 public:
  UniformInplaceGradNode() : egr::GradNodeBase() {}
  UniformInplaceGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~UniformInplaceGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "UniformInplaceGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<UniformInplaceGradNode>(
        new UniformInplaceGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributemin(const float& min) { min_ = min; }
  void SetAttributemax(const float& max) { max_ = max; }
  void SetAttributeseed(const int& seed) { seed_ = seed; }
  void SetAttributediag_num(const int& diag_num) { diag_num_ = diag_num; }
  void SetAttributediag_step(const int& diag_step) { diag_step_ = diag_step; }
  void SetAttributediag_val(const float& diag_val) { diag_val_ = diag_val; }

 private:
  // TensorWrappers

  // Attributes
  float min_;
  float max_;
  int seed_;
  int diag_num_;
  int diag_step_;
  float diag_val_;
};

class UnpoolGradNode : public egr::GradNodeBase {
 public:
  UnpoolGradNode() : egr::GradNodeBase() {}
  UnpoolGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~UnpoolGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "UnpoolGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    indices_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<UnpoolGradNode>(new UnpoolGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperindices(const paddle::experimental::Tensor& indices) {
    indices_ = egr::TensorWrapper(indices, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributeksize(const std::vector<int>& ksize) { ksize_ = ksize; }
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepadding(const std::vector<int>& padding) {
    padding_ = padding;
  }
  void SetAttributeoutput_size(
      const paddle::experimental::IntArray& output_size) {
    output_size_ = output_size;
  }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper indices_;
  egr::TensorWrapper out_;

  // Attributes
  std::vector<int> ksize_;
  std::vector<int> strides_;
  std::vector<int> padding_;
  paddle::experimental::IntArray output_size_;
  std::string data_format_;
};

class Unpool3dGradNode : public egr::GradNodeBase {
 public:
  Unpool3dGradNode() : egr::GradNodeBase() {}
  Unpool3dGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Unpool3dGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Unpool3dGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    indices_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<Unpool3dGradNode>(new Unpool3dGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperindices(const paddle::experimental::Tensor& indices) {
    indices_ = egr::TensorWrapper(indices, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributeksize(const std::vector<int>& ksize) { ksize_ = ksize; }
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepadding(const std::vector<int>& padding) {
    padding_ = padding;
  }
  void SetAttributeoutput_size(const std::vector<int>& output_size) {
    output_size_ = output_size;
  }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper indices_;
  egr::TensorWrapper out_;

  // Attributes
  std::vector<int> ksize_;
  std::vector<int> strides_;
  std::vector<int> padding_;
  std::vector<int> output_size_;
  std::string data_format_;
};

class WarpctcGradNode : public egr::GradNodeBase {
 public:
  WarpctcGradNode() : egr::GradNodeBase() {}
  WarpctcGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~WarpctcGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "WarpctcGradNode"; }

  void ClearTensorWrappers() override {
    logits_.clear();
    logits_length_.clear();
    warpctcgrad_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<WarpctcGradNode>(new WarpctcGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperlogits(const paddle::experimental::Tensor& logits) {
    logits_ = egr::TensorWrapper(logits, true);
  }
  void SetTensorWrapperlogits_length(
      const paddle::experimental::Tensor& logits_length) {
    logits_length_ = egr::TensorWrapper(logits_length, false);
  }
  void SetTensorWrapperwarpctcgrad(
      const paddle::experimental::Tensor& warpctcgrad) {
    warpctcgrad_ = egr::TensorWrapper(warpctcgrad, false);
  }

  // SetAttributes
  void SetAttributeblank(const int& blank) { blank_ = blank; }
  void SetAttributenorm_by_times(const bool& norm_by_times) {
    norm_by_times_ = norm_by_times;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper logits_;
  egr::TensorWrapper logits_length_;
  egr::TensorWrapper warpctcgrad_;

  // Attributes
  int blank_;
  bool norm_by_times_;
};

class YoloLossGradNode : public egr::GradNodeBase {
 public:
  YoloLossGradNode() : egr::GradNodeBase() {}
  YoloLossGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~YoloLossGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "YoloLossGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    gt_box_.clear();
    gt_label_.clear();
    gt_score_.clear();
    objectness_mask_.clear();
    gt_match_mask_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<YoloLossGradNode>(new YoloLossGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappergt_box(const paddle::experimental::Tensor& gt_box) {
    gt_box_ = egr::TensorWrapper(gt_box, false);
  }
  void SetTensorWrappergt_label(const paddle::experimental::Tensor& gt_label) {
    gt_label_ = egr::TensorWrapper(gt_label, false);
  }
  void SetTensorWrappergt_score(const paddle::experimental::Tensor& gt_score) {
    gt_score_ = egr::TensorWrapper(gt_score, false);
  }
  void SetTensorWrapperobjectness_mask(
      const paddle::experimental::Tensor& objectness_mask) {
    objectness_mask_ = egr::TensorWrapper(objectness_mask, false);
  }
  void SetTensorWrappergt_match_mask(
      const paddle::experimental::Tensor& gt_match_mask) {
    gt_match_mask_ = egr::TensorWrapper(gt_match_mask, false);
  }

  // SetAttributes
  void SetAttributeanchors(const std::vector<int>& anchors) {
    anchors_ = anchors;
  }
  void SetAttributeanchor_mask(const std::vector<int>& anchor_mask) {
    anchor_mask_ = anchor_mask;
  }
  void SetAttributeclass_num(const int& class_num) { class_num_ = class_num; }
  void SetAttributeignore_thresh(const float& ignore_thresh) {
    ignore_thresh_ = ignore_thresh;
  }
  void SetAttributedownsample_ratio(const int& downsample_ratio) {
    downsample_ratio_ = downsample_ratio;
  }
  void SetAttributeuse_label_smooth(const bool& use_label_smooth) {
    use_label_smooth_ = use_label_smooth;
  }
  void SetAttributescale_x_y(const float& scale_x_y) { scale_x_y_ = scale_x_y; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper gt_box_;
  egr::TensorWrapper gt_label_;
  egr::TensorWrapper gt_score_;
  egr::TensorWrapper objectness_mask_;
  egr::TensorWrapper gt_match_mask_;

  // Attributes
  std::vector<int> anchors_;
  std::vector<int> anchor_mask_;
  int class_num_;
  float ignore_thresh_;
  int downsample_ratio_;
  bool use_label_smooth_ = true;
  float scale_x_y_ = 1.0;
};

namespace sparse {

class AbsGradNode : public egr::GradNodeBase {
 public:
  AbsGradNode() : egr::GradNodeBase() {}
  AbsGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AbsGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AbsGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AbsGradNode>(new AbsGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class AcosGradNode : public egr::GradNodeBase {
 public:
  AcosGradNode() : egr::GradNodeBase() {}
  AcosGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AcosGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AcosGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AcosGradNode>(new AcosGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class AcoshGradNode : public egr::GradNodeBase {
 public:
  AcoshGradNode() : egr::GradNodeBase() {}
  AcoshGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AcoshGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AcoshGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AcoshGradNode>(new AcoshGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class AddGradNode : public egr::GradNodeBase {
 public:
  AddGradNode() : egr::GradNodeBase() {}
  AddGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AddGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AddGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AddGradNode>(new AddGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
};

class AsinGradNode : public egr::GradNodeBase {
 public:
  AsinGradNode() : egr::GradNodeBase() {}
  AsinGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AsinGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AsinGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AsinGradNode>(new AsinGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class AsinhGradNode : public egr::GradNodeBase {
 public:
  AsinhGradNode() : egr::GradNodeBase() {}
  AsinhGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AsinhGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AsinhGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AsinhGradNode>(new AsinhGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class AtanGradNode : public egr::GradNodeBase {
 public:
  AtanGradNode() : egr::GradNodeBase() {}
  AtanGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AtanGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AtanGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AtanGradNode>(new AtanGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class AtanhGradNode : public egr::GradNodeBase {
 public:
  AtanhGradNode() : egr::GradNodeBase() {}
  AtanhGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AtanhGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AtanhGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AtanhGradNode>(new AtanhGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class BatchNormGradNode : public egr::GradNodeBase {
 public:
  BatchNormGradNode() : egr::GradNodeBase() {}
  BatchNormGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~BatchNormGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "BatchNormGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    scale_.clear();
    bias_.clear();
    mean_out_.clear();
    variance_out_.clear();
    saved_mean_.clear();
    saved_variance_.clear();
    reserve_space_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<BatchNormGradNode>(new BatchNormGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperscale(const paddle::experimental::Tensor& scale) {
    scale_ = egr::TensorWrapper(scale, false);
  }
  void SetTensorWrapperbias(const paddle::experimental::Tensor& bias) {
    bias_ = egr::TensorWrapper(bias, false);
  }
  void SetTensorWrappermean_out(const paddle::experimental::Tensor& mean_out) {
    mean_out_ = egr::TensorWrapper(mean_out, false);
  }
  void SetTensorWrappervariance_out(
      const paddle::experimental::Tensor& variance_out) {
    variance_out_ = egr::TensorWrapper(variance_out, false);
  }
  void SetTensorWrappersaved_mean(
      const paddle::experimental::Tensor& saved_mean) {
    saved_mean_ = egr::TensorWrapper(saved_mean, false);
  }
  void SetTensorWrappersaved_variance(
      const paddle::experimental::Tensor& saved_variance) {
    saved_variance_ = egr::TensorWrapper(saved_variance, false);
  }
  void SetTensorWrapperreserve_space(
      const paddle::experimental::Tensor& reserve_space) {
    reserve_space_ = egr::TensorWrapper(reserve_space, false);
  }

  // SetAttributes
  void SetAttributemomentum(const float& momentum) { momentum_ = momentum; }
  void SetAttributeepsilon(const float& epsilon) { epsilon_ = epsilon; }
  void SetAttributedata_layout(const std::string& data_layout) {
    data_layout_ = data_layout;
  }
  void SetAttributeis_test(const bool& is_test) { is_test_ = is_test; }
  void SetAttributeuse_global_stats(const bool& use_global_stats) {
    use_global_stats_ = use_global_stats;
  }
  void SetAttributetrainable_statistics(const bool& trainable_statistics) {
    trainable_statistics_ = trainable_statistics;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper scale_;
  egr::TensorWrapper bias_;
  egr::TensorWrapper mean_out_;
  egr::TensorWrapper variance_out_;
  egr::TensorWrapper saved_mean_;
  egr::TensorWrapper saved_variance_;
  egr::TensorWrapper reserve_space_;

  // Attributes
  float momentum_;
  float epsilon_;
  std::string data_layout_;
  bool is_test_;
  bool use_global_stats_;
  bool trainable_statistics_;
};

class CastGradNode : public egr::GradNodeBase {
 public:
  CastGradNode() : egr::GradNodeBase() {}
  CastGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~CastGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "CastGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<CastGradNode>(new CastGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributevalue_dtype(
      const paddle::experimental::DataType& value_dtype) {
    value_dtype_ = value_dtype;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  paddle::experimental::DataType value_dtype_;
};

class Conv3dGradNode : public egr::GradNodeBase {
 public:
  Conv3dGradNode() : egr::GradNodeBase() {}
  Conv3dGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Conv3dGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Conv3dGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    kernel_.clear();
    out_.clear();
    rulebook_.clear();
    counter_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<Conv3dGradNode>(new Conv3dGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperkernel(const paddle::experimental::Tensor& kernel) {
    kernel_ = egr::TensorWrapper(kernel, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }
  void SetTensorWrapperrulebook(const paddle::experimental::Tensor& rulebook) {
    rulebook_ = egr::TensorWrapper(rulebook, false);
  }
  void SetTensorWrappercounter(const paddle::experimental::Tensor& counter) {
    counter_ = egr::TensorWrapper(counter, false);
  }

  // SetAttributes
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributedilations(const std::vector<int>& dilations) {
    dilations_ = dilations;
  }
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributegroups(const int& groups) { groups_ = groups; }
  void SetAttributesubm(const bool& subm) { subm_ = subm; }
  void SetAttributekey(const std::string& key) { key_ = key; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper kernel_;
  egr::TensorWrapper out_;
  egr::TensorWrapper rulebook_;
  egr::TensorWrapper counter_;

  // Attributes
  std::vector<int> paddings_;
  std::vector<int> dilations_;
  std::vector<int> strides_;
  int groups_;
  bool subm_;
  std::string key_;
};

class DivideGradNode : public egr::GradNodeBase {
 public:
  DivideGradNode() : egr::GradNodeBase() {}
  DivideGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~DivideGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "DivideGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<DivideGradNode>(new DivideGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;
  egr::TensorWrapper out_;

  // Attributes
};

class DivideScalarGradNode : public egr::GradNodeBase {
 public:
  DivideScalarGradNode() : egr::GradNodeBase() {}
  DivideScalarGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~DivideScalarGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "DivideScalarGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<DivideScalarGradNode>(new DivideScalarGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributescalar(const float& scalar) { scalar_ = scalar; }

 private:
  // TensorWrappers

  // Attributes
  float scalar_;
};

class Expm1GradNode : public egr::GradNodeBase {
 public:
  Expm1GradNode() : egr::GradNodeBase() {}
  Expm1GradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Expm1GradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Expm1GradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<Expm1GradNode>(new Expm1GradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
};

class LeakyReluGradNode : public egr::GradNodeBase {
 public:
  LeakyReluGradNode() : egr::GradNodeBase() {}
  LeakyReluGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~LeakyReluGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "LeakyReluGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<LeakyReluGradNode>(new LeakyReluGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributealpha(const float& alpha) { alpha_ = alpha; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  float alpha_;
};

class Log1pGradNode : public egr::GradNodeBase {
 public:
  Log1pGradNode() : egr::GradNodeBase() {}
  Log1pGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Log1pGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Log1pGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<Log1pGradNode>(new Log1pGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class MultiplyGradNode : public egr::GradNodeBase {
 public:
  MultiplyGradNode() : egr::GradNodeBase() {}
  MultiplyGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MultiplyGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MultiplyGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MultiplyGradNode>(new MultiplyGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
};

class PowGradNode : public egr::GradNodeBase {
 public:
  PowGradNode() : egr::GradNodeBase() {}
  PowGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~PowGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "PowGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<PowGradNode>(new PowGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes
  void SetAttributefactor(const float& factor) { factor_ = factor; }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
  float factor_;
};

class ReluGradNode : public egr::GradNodeBase {
 public:
  ReluGradNode() : egr::GradNodeBase() {}
  ReluGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ReluGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ReluGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<ReluGradNode>(new ReluGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
};

class Relu6GradNode : public egr::GradNodeBase {
 public:
  Relu6GradNode() : egr::GradNodeBase() {}
  Relu6GradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Relu6GradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "Relu6GradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<Relu6GradNode>(new Relu6GradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributethreshold(const float& threshold) { threshold_ = threshold; }

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
  float threshold_ = 6;
};

class ReshapeGradNode : public egr::GradNodeBase {
 public:
  ReshapeGradNode() : egr::GradNodeBase() {}
  ReshapeGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ReshapeGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ReshapeGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ReshapeGradNode>(new ReshapeGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class ScaleGradNode : public egr::GradNodeBase {
 public:
  ScaleGradNode() : egr::GradNodeBase() {}
  ScaleGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ScaleGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ScaleGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<ScaleGradNode>(new ScaleGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributescale(const float& scale) { scale_ = scale; }

 private:
  // TensorWrappers

  // Attributes
  float scale_;
};

class SinGradNode : public egr::GradNodeBase {
 public:
  SinGradNode() : egr::GradNodeBase() {}
  SinGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SinGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SinGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SinGradNode>(new SinGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class SinhGradNode : public egr::GradNodeBase {
 public:
  SinhGradNode() : egr::GradNodeBase() {}
  SinhGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SinhGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SinhGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SinhGradNode>(new SinhGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class SoftmaxGradNode : public egr::GradNodeBase {
 public:
  SoftmaxGradNode() : egr::GradNodeBase() {}
  SoftmaxGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SoftmaxGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SoftmaxGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SoftmaxGradNode>(new SoftmaxGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributeaxis(const int& axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
  int axis_;
};

class SparseCooTensorGradNode : public egr::GradNodeBase {
 public:
  SparseCooTensorGradNode() : egr::GradNodeBase() {}
  SparseCooTensorGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SparseCooTensorGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SparseCooTensorGradNode"; }

  void ClearTensorWrappers() override {
    indices_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SparseCooTensorGradNode>(
        new SparseCooTensorGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperindices(const paddle::experimental::Tensor& indices) {
    indices_ = egr::TensorWrapper(indices, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper indices_;

  // Attributes
};

class SqrtGradNode : public egr::GradNodeBase {
 public:
  SqrtGradNode() : egr::GradNodeBase() {}
  SqrtGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SqrtGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SqrtGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SqrtGradNode>(new SqrtGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
};

class SquareGradNode : public egr::GradNodeBase {
 public:
  SquareGradNode() : egr::GradNodeBase() {}
  SquareGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SquareGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SquareGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SquareGradNode>(new SquareGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class SubtractGradNode : public egr::GradNodeBase {
 public:
  SubtractGradNode() : egr::GradNodeBase() {}
  SubtractGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SubtractGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SubtractGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<SubtractGradNode>(new SubtractGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
};

class SyncBatchNormGradNode : public egr::GradNodeBase {
 public:
  SyncBatchNormGradNode() : egr::GradNodeBase() {}
  SyncBatchNormGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~SyncBatchNormGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "SyncBatchNormGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    scale_.clear();
    bias_.clear();
    saved_mean_.clear();
    saved_variance_.clear();
    reserve_space_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<SyncBatchNormGradNode>(
        new SyncBatchNormGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperscale(const paddle::experimental::Tensor& scale) {
    scale_ = egr::TensorWrapper(scale, false);
  }
  void SetTensorWrapperbias(const paddle::experimental::Tensor& bias) {
    bias_ = egr::TensorWrapper(bias, false);
  }
  void SetTensorWrappersaved_mean(
      const paddle::experimental::Tensor& saved_mean) {
    saved_mean_ = egr::TensorWrapper(saved_mean, false);
  }
  void SetTensorWrappersaved_variance(
      const paddle::experimental::Tensor& saved_variance) {
    saved_variance_ = egr::TensorWrapper(saved_variance, false);
  }
  void SetTensorWrapperreserve_space(
      const paddle::experimental::Tensor& reserve_space) {
    reserve_space_ = egr::TensorWrapper(reserve_space, false);
  }

  // SetAttributes
  void SetAttributemomentum(const float& momentum) { momentum_ = momentum; }
  void SetAttributeepsilon(const float& epsilon) { epsilon_ = epsilon; }
  void SetAttributedata_layout(const std::string& data_layout) {
    data_layout_ = data_layout;
  }
  void SetAttributeis_test(const bool& is_test) { is_test_ = is_test; }
  void SetAttributeuse_global_stats(const bool& use_global_stats) {
    use_global_stats_ = use_global_stats;
  }
  void SetAttributetrainable_statistics(const bool& trainable_statistics) {
    trainable_statistics_ = trainable_statistics;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper scale_;
  egr::TensorWrapper bias_;
  egr::TensorWrapper saved_mean_;
  egr::TensorWrapper saved_variance_;
  egr::TensorWrapper reserve_space_;

  // Attributes
  float momentum_;
  float epsilon_;
  std::string data_layout_;
  bool is_test_;
  bool use_global_stats_;
  bool trainable_statistics_;
};

class TanGradNode : public egr::GradNodeBase {
 public:
  TanGradNode() : egr::GradNodeBase() {}
  TanGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TanGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TanGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<TanGradNode>(new TanGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class TanhGradNode : public egr::GradNodeBase {
 public:
  TanhGradNode() : egr::GradNodeBase() {}
  TanhGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TanhGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TanhGradNode"; }

  void ClearTensorWrappers() override {
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<TanhGradNode>(new TanhGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper out_;

  // Attributes
};

class ToDenseGradNode : public egr::GradNodeBase {
 public:
  ToDenseGradNode() : egr::GradNodeBase() {}
  ToDenseGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ToDenseGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ToDenseGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ToDenseGradNode>(new ToDenseGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class ToSparseCooGradNode : public egr::GradNodeBase {
 public:
  ToSparseCooGradNode() : egr::GradNodeBase() {}
  ToSparseCooGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ToSparseCooGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ToSparseCooGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ToSparseCooGradNode>(new ToSparseCooGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes

 private:
  // TensorWrappers

  // Attributes
};

class TransposeGradNode : public egr::GradNodeBase {
 public:
  TransposeGradNode() : egr::GradNodeBase() {}
  TransposeGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~TransposeGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "TransposeGradNode"; }

  void ClearTensorWrappers() override { SetIsTensorWrappersCleared(true); }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<TransposeGradNode>(new TransposeGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...

  // SetAttributes
  void SetAttributeperm(const std::vector<int>& perm) { perm_ = perm; }

 private:
  // TensorWrappers

  // Attributes
  std::vector<int> perm_;
};

class ValuesGradNode : public egr::GradNodeBase {
 public:
  ValuesGradNode() : egr::GradNodeBase() {}
  ValuesGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~ValuesGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "ValuesGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<ValuesGradNode>(new ValuesGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;

  // Attributes
};

class AddmmGradNode : public egr::GradNodeBase {
 public:
  AddmmGradNode() : egr::GradNodeBase() {}
  AddmmGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AddmmGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AddmmGradNode"; }

  void ClearTensorWrappers() override {
    input_.clear();
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<AddmmGradNode>(new AddmmGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperinput(const paddle::experimental::Tensor& input) {
    input_ = egr::TensorWrapper(input, false);
  }
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes
  void SetAttributealpha(const float& alpha) { alpha_ = alpha; }
  void SetAttributebeta(const float& beta) { beta_ = beta; }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
  float alpha_ = 1.0;
  float beta_ = 1.0;
};

class FusedAttentionGradNode : public egr::GradNodeBase {
 public:
  FusedAttentionGradNode() : egr::GradNodeBase() {}
  FusedAttentionGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~FusedAttentionGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "FusedAttentionGradNode"; }

  void ClearTensorWrappers() override {
    query_.clear();
    key_.clear();
    value_.clear();
    softmax_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<FusedAttentionGradNode>(
        new FusedAttentionGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperquery(const paddle::experimental::Tensor& query) {
    query_ = egr::TensorWrapper(query, false);
  }
  void SetTensorWrapperkey(const paddle::experimental::Tensor& key) {
    key_ = egr::TensorWrapper(key, false);
  }
  void SetTensorWrappervalue(const paddle::experimental::Tensor& value) {
    value_ = egr::TensorWrapper(value, false);
  }
  void SetTensorWrappersoftmax(const paddle::experimental::Tensor& softmax) {
    softmax_ = egr::TensorWrapper(softmax, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper query_;
  egr::TensorWrapper key_;
  egr::TensorWrapper value_;
  egr::TensorWrapper softmax_;

  // Attributes
};

class MaskedMatmulGradNode : public egr::GradNodeBase {
 public:
  MaskedMatmulGradNode() : egr::GradNodeBase() {}
  MaskedMatmulGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MaskedMatmulGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MaskedMatmulGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MaskedMatmulGradNode>(new MaskedMatmulGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
};

class MatmulGradNode : public egr::GradNodeBase {
 public:
  MatmulGradNode() : egr::GradNodeBase() {}
  MatmulGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MatmulGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MatmulGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    y_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MatmulGradNode>(new MatmulGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappery(const paddle::experimental::Tensor& y) {
    y_ = egr::TensorWrapper(y, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper y_;

  // Attributes
};

class MaxpoolGradNode : public egr::GradNodeBase {
 public:
  MaxpoolGradNode() : egr::GradNodeBase() {}
  MaxpoolGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MaxpoolGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MaxpoolGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    rulebook_.clear();
    counter_.clear();
    out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<MaxpoolGradNode>(new MaxpoolGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrapperrulebook(const paddle::experimental::Tensor& rulebook) {
    rulebook_ = egr::TensorWrapper(rulebook, false);
  }
  void SetTensorWrappercounter(const paddle::experimental::Tensor& counter) {
    counter_ = egr::TensorWrapper(counter, false);
  }
  void SetTensorWrapperout(const paddle::experimental::Tensor& out) {
    out_ = egr::TensorWrapper(out, false);
  }

  // SetAttributes
  void SetAttributekernel_sizes(const std::vector<int>& kernel_sizes) {
    kernel_sizes_ = kernel_sizes;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper rulebook_;
  egr::TensorWrapper counter_;
  egr::TensorWrapper out_;

  // Attributes
  std::vector<int> kernel_sizes_;
};

class MvGradNode : public egr::GradNodeBase {
 public:
  MvGradNode() : egr::GradNodeBase() {}
  MvGradNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~MvGradNode() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                                  egr::kSlotSmallVectorSize>& grads,
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "MvGradNode"; }

  void ClearTensorWrappers() override {
    x_.clear();
    vec_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<MvGradNode>(new MvGradNode(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const paddle::experimental::Tensor& x) {
    x_ = egr::TensorWrapper(x, false);
  }
  void SetTensorWrappervec(const paddle::experimental::Tensor& vec) {
    vec_ = egr::TensorWrapper(vec, false);
  }

  // SetAttributes

 private:
  // TensorWrappers
  egr::TensorWrapper x_;
  egr::TensorWrapper vec_;

  // Attributes
};

}  // namespace sparse

namespace strings {}
