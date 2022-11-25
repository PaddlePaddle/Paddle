// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/operators/reduce_ops/reduce_min_max_op.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace ops = paddle::operators;

class ReduceMinOpMaker : public ops::ReduceOpMaker {
 protected:
  virtual std::string GetName() const { return "reduce_min"; }
  virtual std::string GetOpType() const { return "Reduce reduce_min"; }
};

<<<<<<< HEAD
DECLARE_INFER_SHAPE_FUNCTOR(
    reduce_min,
    ReduceMinInferShapeFunctor,
    PD_INFER_META(phi::ReduceIntArrayAxisInferMetaBase));
=======
DECLARE_INFER_SHAPE_FUNCTOR(reduce_min,
                            ReduceMinInferShapeFunctor,
                            PD_INFER_META(phi::ReduceInferMetaBase));
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

REGISTER_OPERATOR(
    reduce_min,
    ops::ReduceOp,
    ReduceMinOpMaker,
    paddle::framework::DefaultGradOpMaker<paddle::framework::OpDesc, true>,
    paddle::framework::DefaultGradOpMaker<paddle::imperative::OpBase, true>,
    ReduceMinInferShapeFunctor);
REGISTER_OPERATOR(reduce_min_grad, ops::ReduceGradOp)
