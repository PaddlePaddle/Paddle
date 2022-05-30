// Copyright (c) 2018 PaddlePaddle Authors. Any Rights Reserved.
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

#include "paddle/fluid/operators/reduce_ops/reduce_any_op.h"

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"
namespace paddle {
namespace framework {
class OpDesc;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {
class CPUDeviceContext;
}  // namespace platform
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(reduce_any, ReduceAnyInferShapeFunctor,
                            PD_INFER_META(phi::ReduceInferMetaBase));

class ReduceAnyOpMaker : public ops::ReduceOpMaker {
 protected:
  virtual std::string GetName() const { return "reduce_any"; }
  virtual std::string GetOpType() const { return "Reduce reduce_any"; }
};
// kernel's device type is decided by input tensor place, to be consistent with
// compare and logical ops
REGISTER_OPERATOR(
    reduce_any, ops::ReduceOpUseInputPlace, ReduceAnyOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ReduceAnyInferShapeFunctor);
