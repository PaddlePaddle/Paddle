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

#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/pir/core/op_base.h"

namespace paddle {
namespace dialect {
class InferMetaInterface : public pir::OpInterfaceBase<InferMetaInterface> {
 public:
  /// Defined these methods with the interface.
  struct Concept {
    explicit Concept(void (*infer_meta)(phi::InferMetaContext *))
        : infer_meta_(infer_meta) {}
    void (*infer_meta_)(phi::InferMetaContext *);
  };

  template <class ConcreteOp>
  struct Model : public Concept {
    static inline void InferMeta(phi::InferMetaContext *infer_meta) {
      return ConcreteOp::InferMeta(infer_meta);
    }

    Model() : Concept(InferMeta) {}
  };

  /// Constructor
  InferMetaInterface(pir::Operation *op, Concept *impl)
      : pir::OpInterfaceBase<InferMetaInterface>(op), impl_(impl) {}

  void InferMeta(phi::InferMetaContext *infer_meta) {
    impl_->infer_meta_(infer_meta);
  }

 private:
  Concept *impl_;
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::InferMetaInterface)
