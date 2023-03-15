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
#include "paddle/fluid/prim/api/composite_backward/composite_backward_api.h"
#include "paddle/fluid/prim/utils/static/composite_grad_desc_maker.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace ops = paddle::operators;

class ReduceMaxOpMaker : public ops::ReduceOpMaker {
 protected:
  virtual std::string GetName() const { return "reduce_max"; }
  virtual std::string GetOpType() const { return "Reduce reduce_max"; }
};

namespace paddle {
namespace operators {

class ReduceMaxCompositeGradOpMaker : public prim::CompositeGradOpMakerBase {
 public:
  using prim::CompositeGradOpMakerBase::CompositeGradOpMakerBase;
  void Apply() override {
    paddle::Tensor x = this->GetSingleForwardInput("X");
    paddle::Tensor out = this->GetSingleForwardInput("Out");
    paddle::Tensor out_grad = this->GetSingleOutputGrad("Out");
    std::vector<int> axis = this->Attr<std::vector<int>>("dim");
    bool keep_dim = this->Attr<bool>("keep_dim");
    bool reduce_all = this->Attr<bool>("reduce_all");
    paddle::Tensor x_grad_t = this->GetSingleInputGrad("X");
    paddle::Tensor* x_grad = this->GetOutputPtr(&x_grad_t);
    std::string x_grad_name = this->GetOutputName(x_grad_t);
    VLOG(6) << "Runing max_grad composite func";
    prim::max_grad<prim::DescTensor>(
        x, out, out_grad, axis, keep_dim, reduce_all, x_grad);
    this->RecoverOutputName(x_grad_t, x_grad_name);
  }
};

}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(
    reduce_max,
    ReduceMaxInferShapeFunctor,
    PD_INFER_META(phi::ReduceIntArrayAxisInferMetaBase));

REGISTER_OPERATOR(
    reduce_max,
    ops::ReduceOp,
    ReduceMaxOpMaker,
    paddle::framework::DefaultGradOpMaker<paddle::framework::OpDesc, true>,
    paddle::framework::DefaultGradOpMaker<paddle::imperative::OpBase, true>,
    ops::ReduceMaxCompositeGradOpMaker,
    ReduceMaxInferShapeFunctor);
REGISTER_OPERATOR(reduce_max_grad, ops::ReduceGradOp)
