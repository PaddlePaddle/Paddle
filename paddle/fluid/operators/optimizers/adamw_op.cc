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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/operators/optimizers/adam_op.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

class AdamWOp : public AdamOp {
  using AdamOp::AdamOp;
};

class AdamWOpMaker : public AdamOpMaker {
 public:
  void Make() {
    AdamOpMaker::Make();
    AddAttr<float>("lr_ratio",
                   "(float, default 1.0) "
                   "layerwise learning rate decay")
        .SetDefault(1.0f);
    AddAttr<float>("coeff",
                   "(float, default 0.01) "
                   "coeff of the weight decay")
        .SetDefault(0.01f);
    AddAttr<bool>("with_decay",
                  "(bool, default false) "
                  "whether to do weight decay")
        .SetDefault(false);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(adamw,
                            AdamwInferMetaFunctor,
                            PD_INFER_META(phi::AdamwInferMeta));
REGISTER_OPERATOR(
    adamw,
    ops::AdamWOp,
    ops::AdamWOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    AdamwInferMetaFunctor);
