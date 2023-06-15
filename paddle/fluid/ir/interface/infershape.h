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

#include "paddle/ir/core/op_base.h"
#include "paddle/phi/core/infermeta_utils.h"

namespace paddle {
namespace dialect {
class InferShapeInterface : public ir::OpInterfaceBase<InferShapeInterface> {
 public:
  struct Concept {
    explicit Concept(void (*infer_shape)(phi::InferMetaContext *))
        : infer_shape_(infer_shape) {}
    void (*infer_shape_)(phi::InferMetaContext *);
  };

  template <class ConcreteOp>
  struct Model : public Concept {
    static void InferShape(phi::InferMetaContext *infer_meta) {
      return ConcreteOp::InferShape(infer_meta);
    }

    Model() : Concept(InferShape) {}
  };

  InferShapeInterface(ir::Operation *op, Concept *impl)
      : ir::OpInterfaceBase<InferShapeInterface>(op), impl_(impl) {}

  void InferShape(phi::InferMetaContext *infer_meta) {
    impl_->infer_shape_(infer_meta);
  }

 private:
  Concept *impl_;
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(InferShapeInterface)
