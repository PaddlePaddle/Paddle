/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/operators/uniform_random_op.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/batch_size_like.h"

namespace paddle {
namespace operators {

namespace {
template <typename T>
inline void UniformRealDistribution(T *data,
                                    const int64_t &size,
                                    const float &min,
                                    const float &max,
                                    const unsigned int seed) {
  VLOG(4) << "[CPU] UniformRandomKernel<T>";
  std::uniform_real_distribution<T> dist(static_cast<T>(min),
                                         static_cast<T>(max));
  auto engine = phi::GetCPURandomEngine(seed);

  for (int64_t i = 0; i < size; ++i) {
    data[i] = dist(*engine);
  }
}

template <>
inline void UniformRealDistribution(paddle::platform::bfloat16 *data,
                                    const int64_t &size,
                                    const float &min,
                                    const float &max,
                                    const unsigned int seed) {
  VLOG(4) << "[CPU] UniformRandomKernel<bfloat16>";
  std::uniform_real_distribution<float> dist(min, max);
  auto engine = phi::GetCPURandomEngine(seed);

  for (int64_t i = 0; i < size; ++i) {
    data[i] = static_cast<paddle::platform::bfloat16>(dist(*engine));
  }
}
}  // namespace

// It seems that Eigen::Tensor::random in GPU will SEGFAULT.
// Use std::random and thrust::random(thrust is a std library in CUDA) to
// implement uniform random.
template <typename T, typename DeviceContext>
class CPUUniformRandomKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    phi::DenseTensor *tensor = nullptr;
    auto out_var = ctx.OutputVar("Out");
    std::vector<int64_t> new_shape;
    auto list_new_shape_tensor =
        ctx.MultiInput<phi::DenseTensor>("ShapeTensorList");
    if (!list_new_shape_tensor.empty() || ctx.HasInput("ShapeTensor")) {
      if (ctx.HasInput("ShapeTensor")) {
        auto *shape_tensor = ctx.Input<phi::DenseTensor>("ShapeTensor");
        new_shape = GetNewDataFromShapeTensor(shape_tensor);
      } else if (!list_new_shape_tensor.empty()) {
        new_shape = GetNewDataFromShapeTensorList(list_new_shape_tensor);
      }
    }

    if (out_var->IsType<phi::SelectedRows>()) {
      auto *selected_rows = out_var->GetMutable<phi::SelectedRows>();
      tensor = selected_rows->mutable_value();
      auto shape = ctx.Attr<std::vector<int64_t>>("shape");
      if (!new_shape.empty()) shape = new_shape;
      tensor->Resize(phi::make_ddim(shape));
      selected_rows->mutable_rows()->reserve(shape[0]);
    } else if (out_var->IsType<phi::DenseTensor>()) {
      tensor = out_var->GetMutable<phi::DenseTensor>();
      if (!new_shape.empty()) tensor->Resize(phi::make_ddim(new_shape));
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Expected type of Output(out) in uniform_random_op must be Tensor, "
          "SelectedRows. But got "
          "unsupport type: %s.",
          framework::ToTypeName(out_var->Type())));
    }
    T *data = tensor->mutable_data<T>(ctx.GetPlace());
    int64_t size = tensor->numel();

    UniformRealDistribution<T>(
        data,
        size,
        ctx.Attr<float>("min"),
        ctx.Attr<float>("max"),
        static_cast<unsigned int>(ctx.Attr<int>("seed")));

    unsigned int diag_num =
        static_cast<unsigned int>(ctx.Attr<int>("diag_num"));
    unsigned int diag_step =
        static_cast<unsigned int>(ctx.Attr<int>("diag_step"));
    auto diag_val = static_cast<T>(ctx.Attr<float>("diag_val"));
    if (diag_num > 0) {
      PADDLE_ENFORCE_GT(
          size,
          (diag_num - 1) * (diag_step + 1),
          platform::errors::InvalidArgument(
              "ShapeInvalid: the diagonal's elements is equal (num-1) "
              "* (step-1) with num %d, step %d,"
              "It should be smaller than %d, but received %d",
              diag_num,
              diag_step,
              (diag_num - 1) * (diag_step + 1),
              size));
      for (int64_t i = 0; i < diag_num; ++i) {
        int64_t pos = i * diag_step + i;
        data[pos] = diag_val;
      }
    }
  }
};
class UniformRandomBatchSizeLikeOp : public BatchSizeLikeOp {
 protected:
  using BatchSizeLikeOp::BatchSizeLikeOp;

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return phi::KernelKey(
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }
};

class UniformRandomBatchSizeLikeOpMaker : public BatchSizeLikeOpMaker {
 protected:
  void Apply() override {
    AddComment(R"DOC(
UniformRandomBatchSizeLike operator.

This operator initializes a tensor with the same batch_size as the Input tensor
with random values sampled from a uniform distribution.

)DOC");
    AddAttr<float>("min",
                   "(float, default -1.0) "
                   "Minimum value of uniform random")
        .SetDefault(-1.0f);
    AddAttr<float>("max",
                   "(float, default 1.0) "
                   "Maximun value of uniform random")
        .SetDefault(1.0f);
    AddAttr<int>("seed",
                 "(int, default 0) "
                 "Random seed used for generating samples. "
                 "0 means use a seed generated by the system."
                 "Note that if seed is not 0, this operator will always "
                 "generate the same random numbers every time.")
        .SetDefault(0);
    AddAttr<int>("diag_num",
                 "The number of diag elements. Note that if "
                 "diag_num is 0, it means without diag init.[default 0].")
        .SetDefault(0);
    AddAttr<int>("diag_step", "The step between two diag element.[default 0].")
        .SetDefault(0);
    AddAttr<float>("diag_val", "The value of diag element. [default 1.0].")
        .SetDefault(1.0f);
    AddAttr<int>("dtype", "(int, default 5(FP32)) Output tensor data type")
        .SetDefault(framework::proto::VarType::FP32);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OPERATOR(
    uniform_random_batch_size_like,
    ops::UniformRandomBatchSizeLikeOp,
    ops::UniformRandomBatchSizeLikeOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::BatchSizeLikeNoNeedBufferVarsInferer);

PD_REGISTER_STRUCT_KERNEL(uniform_random_batch_size_like,
                          CPU,
                          ALL_LAYOUT,
                          ops::CPUUniformRandomKernel,
                          float,
                          double,
                          plat::bfloat16) {}
