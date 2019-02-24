// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

static framework::proto::VarType::Type kDefaultDtype =
    static_cast<framework::proto::VarType::Type>(0);

template <typename DeviceContext, typename T>
class AllocContinuousSpaceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto &in_var_names = context.Inputs("Input");
    auto &out_var_names = context.Outputs("Output");

    auto &in_vars = context.MultiInputVar("Input");
    PADDLE_ENFORCE_GT(in_var_names.size(), static_cast<size_t>(0));
    PADDLE_ENFORCE_EQ(in_var_names.size(), out_var_names.size());

    for (size_t i = 0; i < in_var_names.size(); ++i) {
      PADDLE_ENFORCE_EQ(in_var_names[i], out_var_names[i]);
      // Only support LoDTensor,
      PADDLE_ENFORCE(in_vars[i]->IsType<framework::LoDTensor>());
    }

    auto out_tensors = context.MultiOutput<framework::LoDTensor>("Output");
    PADDLE_ENFORCE_EQ(in_var_names.size(), out_tensors.size());

    size_t mem_size = 0;
    auto dtype = kDefaultDtype;
    GetMemSizeAndDtype(out_tensors, out_var_names, &mem_size, &dtype);

    auto fused_tensor = context.Output<framework::LoDTensor>("FusedOutput");
    fused_tensor->Resize(framework::make_ddim({static_cast<int64_t>(mem_size)}))
        .mutable_data(context.GetPlace(), dtype);

    auto &dev_ctx = context.template device_context<DeviceContext>();

    int64_t offset = 0;
    if (context.Attr<bool>("copy_data")) {
      for (size_t i = 0; i < in_var_names.size(); ++i) {
        int64_t len = out_tensors[i]->numel();
        auto sub_tensor = fused_tensor->Slice(offset, offset + len);
        offset += len;
        framework::TensorCopy(*out_tensors[i], context.GetPlace(), dev_ctx,
                              &sub_tensor);
      }
    } else {
      math::SetConstant<DeviceContext, T> set_constant;
      set_constant(dev_ctx, fused_tensor,
                   static_cast<T>(context.Attr<float>("constant")));
    }

    offset = 0;
    for (size_t i = 0; i < out_tensors.size(); ++i) {
      int64_t len = out_tensors[i]->numel();
      auto dim = out_tensors[i]->dims();
      out_tensors[i]
          ->ShareDataWith(fused_tensor->Slice(offset, offset + len))
          .Resize(dim);
      offset += len;
      VLOG(10) << "alloc_space_for_vars: output(" << in_var_names[i]
               << ") ,dim:(" << dim << ")"
               << " Address: " << out_tensors[i]->data<void>();
    }
  }

  void GetMemSizeAndDtype(
      const std::vector<framework::LoDTensor *> &lod_tensors,
      const std::vector<std::string> var_names, size_t *mem_size,
      framework::proto::VarType::Type *dtype) const {
    PADDLE_ENFORCE_EQ(lod_tensors.size(), var_names.size());
    *mem_size = 0;
    for (size_t i = 0; i < var_names.size(); ++i) {
      PADDLE_ENFORCE(lod_tensors[i]->IsInitialized(), "%s is not initialized.",
                     var_names[i]);

      auto p_dtype = lod_tensors[i]->type();
      if (*dtype == kDefaultDtype) {
        PADDLE_ENFORCE_NE(p_dtype, kDefaultDtype, "%s's type should not be %s.",
                          var_names[i], kDefaultDtype);
        *dtype = p_dtype;
      }
      PADDLE_ENFORCE_EQ(p_dtype, *dtype, "Input vars is not equal.");

      auto size = lod_tensors[i]->numel();
      PADDLE_ENFORCE_GT(size, 0);
      VLOG(10) << "alloc_space_for_vars: input(" << var_names[i] << ") ,dim:("
               << lod_tensors[i]->dims() << ")";
      *mem_size += size;
    }
  }
};

class AllocContinuousSpaceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {}
};

class AllocContinuousSpaceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "A set of variables.").AsDuplicable();
    AddOutput("Output", "A set of variables.").AsDuplicable();
    AddOutput("FusedOutput", "");
    AddAttr<bool>("copy_data", ".").SetDefault(false);
    AddAttr<float>("constant", ".").SetDefault(0.0);
    AddComment(R"DOC(
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(alloc_continuous_space,
                  paddle::operators::AllocContinuousSpaceOp,
                  paddle::operators::AllocContinuousSpaceOpMaker);
namespace ops = paddle::operators;
REGISTER_OP_CPU_KERNEL(
    alloc_continuous_space,
    ops::AllocContinuousSpaceKernel<paddle::platform::CPUDeviceContext, int>,
    ops::AllocContinuousSpaceKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AllocContinuousSpaceKernel<paddle::platform::CPUDeviceContext,
                                    double>);

#ifdef PADDLE_WITH_CUDA
REGISTER_OP_CUDA_KERNEL(
    alloc_continuous_space,
    ops::AllocContinuousSpaceKernel<paddle::platform::CUDADeviceContext, int>,
    ops::AllocContinuousSpaceKernel<paddle::platform::CUDADeviceContext, float>,
    ops::AllocContinuousSpaceKernel<paddle::platform::CUDADeviceContext,
                                    double>);
#endif
