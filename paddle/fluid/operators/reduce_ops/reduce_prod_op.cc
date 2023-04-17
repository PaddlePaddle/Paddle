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

#include "paddle/fluid/operators/reduce_ops/reduce_prod_op.h"
#include "paddle/fluid/prim/api/composite_backward/composite_backward_api.h"
#include "paddle/fluid/prim/utils/static/composite_grad_desc_maker.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace framework {
class OpDesc;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {
class ReduceProdCompositeGradOpMaker : public prim::CompositeGradOpMakerBase {
 public:
  using prim::CompositeGradOpMakerBase::CompositeGradOpMakerBase;
  void Apply() override {
    // get inputs
    paddle::Tensor x = this->GetSingleForwardInput("X");
    paddle::Tensor out = this->GetSingleForwardOutput("Out");
    paddle::Tensor out_grad = this->GetSingleOutputGrad("Out");

    // get attr
    std::vector<int> axis = this->Attr<std::vector<int>>("dim");
    bool keep_dim = this->Attr<bool>("keep_dim");
    bool reduce_all = this->Attr<bool>("reduce_all");

    // get output
    paddle::Tensor x_grad_t = this->GetSingleInputGrad("X");

    // get output ptr
    auto x_grad = this->GetOutputPtr(&x_grad_t);

    // get output orginal name
    std::string x_grad_name = this->GetOutputName(x_grad_t);
    VLOG(6) << "Runing prod_grad composite func";
    // call composite backward func
    prim::prod_grad<prim::DescTensor>(
        x, out, out_grad, axis, keep_dim, reduce_all, x_grad);
    // recover output name
    this->RecoverOutputName(x_grad_t, x_grad_name);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

class ReduceProdOpMaker : public ops::ReduceBaseOpMaker {
 protected:
  virtual std::string GetName() const { return "reduce_prod"; }
  virtual std::string GetOpType() const { return "Reduce reduce_prod"; }
};

DECLARE_INFER_SHAPE_FUNCTOR(
    reduce_prod,
    ReduceProdInferShapeFunctor,
    PD_INFER_META(phi::ReduceIntArrayAxisInferMetaBase));

REGISTER_OPERATOR(
    reduce_prod,
    ops::ReduceBaseOp,
    ReduceProdOpMaker,
    paddle::framework::DefaultGradOpMaker<paddle::framework::OpDesc, true>,
    paddle::framework::DefaultGradOpMaker<paddle::imperative::OpBase, true>,
    ops::ReduceProdCompositeGradOpMaker,
    ReduceProdInferShapeFunctor);
REGISTER_OPERATOR(reduce_prod_grad, ops::ReduceGradOp);
