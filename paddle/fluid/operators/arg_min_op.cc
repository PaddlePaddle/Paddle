/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/arg_min_max_op_base.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

DECLARE_INFER_SHAPE_FUNCTOR(arg_min, ArgMinInferShapeFunctor,
                            PD_INFER_META(phi::ArgMinMaxInferMeta));

REGISTER_OPERATOR(
    arg_min, paddle::operators::ArgMinMaxOp, paddle::operators::ArgMinOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ArgMinInferShapeFunctor);

REGISTER_OP_VERSION(arg_min)
    .AddCheckpoint(
        R"ROC(
              Upgrade argmin add a new attribute [flatten] and modify the attribute of dtype)ROC",
        paddle::framework::compatible::OpVersionDesc()
            .NewAttr("flatten",
                     "In order to compute the argmin over the flattened array "
                     "when the "
                     "argument `axis` in python API is None.",
                     false)
            .ModifyAttr(
                "dtype",
                "Change the default value of dtype from -1 to 3"
                ", means return the int64 indices directly. The rearse why "
                "changing the default value is that the int64 value in "
                "VarType is 3 in the frameworke.proto.",
                3));
