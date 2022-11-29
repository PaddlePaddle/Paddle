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

#include "paddle/fluid/operators/clip_by_norm_op.h"
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(clip_by_norm,
                            ClipByNormInferShapeFunctor,
                            PD_INFER_META(phi::ClipByNormInferMeta));

REGISTER_OP_WITHOUT_GRADIENT(clip_by_norm,
                             ops::ClipByNormOp,
                             ops::ClipByNormOpMaker,
                             ClipByNormInferShapeFunctor);
