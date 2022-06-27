/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/optimizers/adam_op.h"

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(adam,
                            AdamInferMetaFunctor,
                            PD_INFER_META(phi::AdamInferMeta));

REGISTER_OPERATOR(
    adam,
    ops::AdamOp,
    ops::AdamOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    AdamInferMetaFunctor);

REGISTER_OP_VERSION(adam)
    .AddCheckpoint(
        R"ROC(
      Upgrade adam add 1 attribute [multi_precision].
    )ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "multi_precision",
            "(bool) Whether to use multi-precision during weight updating.",
            false))
    .AddCheckpoint(
        R"ROC(
      Upgrade adam, add 1 dispensable input [EpsilonTensor].
    )ROC",
        paddle::framework::compatible::OpVersionDesc().NewInput(
            "EpsilonTensor",
            "If provided, Adam will use this as epsilon, "
            "this has a higher priority than attr(epsilon). "
            "For better performance in npu kernel. "))
    .AddCheckpoint(
        R"ROC(
      Upgrade adam, add 1 attribute [use_global_beta_pow].
    )ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "use_global_beta_pow",
            "If true, Adam will use global beta_pow for whole model "
            "instead of creating beta_pow for each parameter."
            "In that case, the outputs(Beta1PowOut, Beta2PowOut) will not be "
            "used in adam op, "
            "and beta_pow will be updated after all adam op in the model.",
            false))
    .AddCheckpoint(
        R"ROC(
      Upgrade adam, add 1 dispensable input [SkipUpdate].
    )ROC",
        paddle::framework::compatible::OpVersionDesc().NewInput(
            "SkipUpdate", "If the value is true, Adam will skip the update."));
