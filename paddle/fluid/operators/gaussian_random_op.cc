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

#include <random>

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/fill_constant_op.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif
#include "paddle/phi/infermeta/nullary.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class CPUGaussianRandomBatchSizeLikeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    float mean = context.Attr<float>("mean");
    float std = context.Attr<float>("std");
    auto* tensor = context.Output<framework::Tensor>("Out");
    T* data = tensor->mutable_data<T>(context.GetPlace());

    unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));
    std::minstd_rand engine;
    if (seed == 0) {
      seed = std::random_device()();
    }
    engine.seed(seed);
    std::normal_distribution<T> dist(mean, std);
    int64_t size = tensor->numel();
    for (int64_t i = 0; i < size; ++i) {
      data[i] = dist(engine);
    }
  }
};

class GaussianRandomOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library{framework::LibraryType::kPlain};
    framework::DataLayout layout{framework::DataLayout::kAnyLayout};
    auto data_type =
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));

#ifdef PADDLE_WITH_MKLDNN
    if (library == framework::LibraryType::kPlain &&
        this->CanMKLDNNBeUsed(ctx, data_type)) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;
    }
#endif

    return framework::OpKernelType(data_type, ctx.device_context(), layout,
                                   library);
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "ShapeTensor" || var_name == "ShapeTensorList") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class GaussianRandomOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "Output matrix of gaussian random op");

    AddAttr<std::vector<int64_t>>("shape",
                                  "(vector<int64_t>) "
                                  "The dimension of random tensor.")
        .SetDefault({});
    AddInput("ShapeTensor",
             "(Tensor<int>), optional). The shape of the output."
             "It has a higher priority than Attr(shape).")
        .AsDispensable();
    AddInput("ShapeTensorList",
             "(vector<Tensor<int>>, optional). The shape of the output. "
             "It has a higher priority than Attr(shape)."
             "The shape of the element in vector must be [1].")
        .AsDuplicable()
        .AsDispensable();
    AddAttr<float>("mean",
                   "(float, default 0.0) "
                   "mean of random tensor.")
        .SetDefault(.0f);
    AddAttr<float>("std",
                   "(float, default 1.0) "
                   "std of random tensor.")
        .SetDefault(1.0f);
    AddAttr<int>("seed",
                 "(int, default 0) "
                 "Random seed of generator."
                 "0 means use system wide seed."
                 "Note that if seed is not 0, this operator will always "
                 "generate the same random numbers every time.")
        .SetDefault(0);
    AddAttr<int>("dtype",
                 "(int, default 5(FP32)) "
                 "Output data type.")
        .SetDefault(framework::proto::VarType::FP32);
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddComment(R"DOC(
GaussianRandom Operator.

Used to initialize tensors with gaussian random generator.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(gaussian_random, GaussianRandomInferShapeFunctor,
                            PD_INFER_META(phi::GaussianRandomInferMeta));

REGISTER_OPERATOR(
    gaussian_random, ops::GaussianRandomOp, ops::GaussianRandomOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    GaussianRandomInferShapeFunctor);

REGISTER_OP_CPU_KERNEL(gaussian_random_batch_size_like,
                       ops::CPUGaussianRandomBatchSizeLikeKernel<float>,
                       ops::CPUGaussianRandomBatchSizeLikeKernel<double>);

REGISTER_OP_VERSION(gaussian_random)
    .AddCheckpoint(
        R"ROC(
               Upgrade gaussian_random add new inputs [ShapeTensor] and [ShapeTensorList] 
               and modify the attribute of [shape])ROC",
        paddle::framework::compatible::OpVersionDesc()
            .NewInput("ShapeTensor",
                      "The output shape supports Tensor type. ShapeTensor is "
                      "dispensable.")
            .NewInput("ShapeTensorList",
                      "The output shape supports list filled with Tensor. "
                      "ShapeTensorList is dispensable.")
            .ModifyAttr("shape",
                        "The arg 'default_value' of attr 'shape' is changed: "
                        "from 'None' to '{}'.",
                        std::vector<int64_t>{}));
