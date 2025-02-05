// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/dialect/operator/interface/layout_transformation.hpp"

#include "paddle/common/enforce.h"
#include "paddle/common/layout.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/type_name.h"

namespace paddle {
namespace dialect {

class LayoutTransformationInterface
    : public pir::OpInterfaceBase<LayoutTransformationInterface> {
 public:
  using PreferLayoutFn = common::DataLayout (*)(pir::Operation*);
  using RewriteByLayoutFn = void (*)(pir::Operation*, common::DataLayout);
  using RelevantInputsFn = std::vector<pir::Value> (*)(pir::Operation*);
  using RelevantOutputsFn = std::vector<pir::Value> (*)(pir::Operation*);
  using CanBeModifiedFn = bool (*)(pir::Operation*);

  struct Concept {
    explicit Concept(PreferLayoutFn prefer_layout,
                     RewriteByLayoutFn rewrite_by_layout,
                     RelevantInputsFn relevant_inputs,
                     RelevantOutputsFn relevant_outputs,
                     CanBeModifiedFn can_be_modified)
        : prefer_layout(prefer_layout),
          rewrite_by_layout(rewrite_by_layout),
          relevant_inputs(relevant_inputs),
          relevant_outputs(relevant_outputs),
          can_be_modified(can_be_modified) {}

    PreferLayoutFn prefer_layout;
    RewriteByLayoutFn rewrite_by_layout;
    RelevantInputsFn relevant_inputs;
    RelevantOutputsFn relevant_outputs;
    CanBeModifiedFn can_be_modified;
  };

  template <typename ConcreteOp>
  struct Model : public Concept {
    static common::DataLayout PreferLayoutModel(pir::Operation* op) {
      return PreferLayoutImpl<ConcreteOp>(op);
    }

    static void RewriteByLayoutModel(pir::Operation* op,
                                     common::DataLayout new_layout) {
      RewriteByLayoutImpl<ConcreteOp>(op, new_layout);
    }

    static std::vector<pir::Value> RelevantInputsModel(pir::Operation* op) {
      return RelevantInputsImpl<ConcreteOp>(op);
    }

    static std::vector<pir::Value> RelevantOutputsModel(pir::Operation* op) {
      return RelevantOutputsImpl<ConcreteOp>(op);
    }

    static bool CanBeModifiedModel(pir::Operation* op) {
      return CanBeModifiedImpl<ConcreteOp>(op);
    }

    Model()
        : Concept(PreferLayoutModel,
                  RewriteByLayoutModel,
                  RelevantInputsModel,
                  RelevantOutputsModel,
                  CanBeModifiedModel) {}
  };

  LayoutTransformationInterface(const pir::Operation* op, Concept* impl)
      : pir::OpInterfaceBase<LayoutTransformationInterface>(op), impl_(impl) {}

  common::DataLayout PreferLayout(pir::Operation* op) {
    return impl_->prefer_layout(op);
  }

  void RewriteByLayout(pir::Operation* op, common::DataLayout new_layout) {
    impl_->rewrite_by_layout(op, new_layout);
  }

  std::vector<pir::Value> RelevantInputs(pir::Operation* op) {
    return impl_->relevant_inputs(op);
  }

  std::vector<pir::Value> RelevantOutputs(pir::Operation* op) {
    return impl_->relevant_outputs(op);
  }

  bool CanBeModified(pir::Operation* op) { return impl_->can_be_modified(op); }

 private:
  Concept* impl_;
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::LayoutTransformationInterface)
