/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/dgc_clip_by_norm_op.h"

#include <string>

namespace paddle {
namespace operators {

class DGCClipByNormOp : public ClipByNormOp {
 public:
  using ClipByNormOp::ClipByNormOp;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("current_step"),
                   "Input",
                   "current_step",
                   "DGCClipByNormOp");

    return ClipByNormOp::InferShape(ctx);
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "current_step") {
      VLOG(10) << "var_name:" << var_name << " need not to transform";
      return expected_kernel_type;
    }

    return framework::OperatorWithKernel::GetKernelTypeForVar(
        var_name, tensor, expected_kernel_type);
  }
};

class DGCClipByNormOpMaker : public ClipByNormOpMaker {
 public:
  void Make() override {
    AddInput("current_step", "(Tensor) Current step.");
    AddAttr<float>("rampup_begin_step",
                   "(float, -1.0)"
                   "The period when begin k_select.")
        .SetDefault(-1.0);

    return ClipByNormOpMaker::Make();
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(dgc_clip_by_norm,
                             ops::DGCClipByNormOp,
                             ops::DGCClipByNormOpMaker);

REGISTER_OP_CPU_KERNEL(dgc_clip_by_norm,
                       ops::DGCClipByNormKernel<phi::CPUContext, float>);
