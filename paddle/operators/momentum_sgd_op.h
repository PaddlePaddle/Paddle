#pragma once
#include <glog/logging.h>
#include <paddle/framework/operator.h>
#include <paddle/framework/tensor.h>

namespace paddle {
namespace operators {

inline void momentum_sgd_update(int N,
                                const float* g,
                                const float* m,
                                float* ng,
                                float* nm,
                                const float* lr,
                                float momentum,
                                bool nesterov,
                                float* param) {
  const float LR = lr[0];
  for (auto i = 0; i < N; ++i) {
    if (!nesterov) {
      const float adjusted_gradient = LR * g[i] + momentum * m[i];
      nm[i] = adjusted_gradient;
      ng[i] = adjusted_gradient;
    } else {
      const float mi = m[i];
      const float mi_new = momentum * mi + LR * g[i];
      nm[i] = mi_new;
      ng[i] = (1 + momentum) * mi_new - momentum * mi;
    }

    if (param) {
      param[i] -= ng[i];
    }
  }
}

template <typename T, typename Place>
class MomentumSGDOpKernel : public framework::OpKernel {
public:
  void Compute(const framework::KernelContext& ctx) const override {
    LOG(INFO) << "Add kernel in " << typeid(Place).name();
    const framework::Tensor& param =
        ctx.Input("param")->Get<framework::Tensor>();
    int size = (int)product(param.dims());

    const framework::Tensor& grad = ctx.Input("grad")->Get<framework::Tensor>();
    const framework::Tensor& moment =
        ctx.Input("moment")->Get<framework::Tensor>();

    const framework::Tensor& output_grad =
        ctx.Input("output_grad")->Get<framework::Tensor>();
    const framework::Tensor& output_moment =
        ctx.Input("output_moment")->Get<framework::Tensor>();

    const framework::Tensor& output_param =
        ctx.Input("output_param")->Get<framework::Tensor>();

    const framework::Tensor& learning_rate =
        ctx.Input("lr")->Get<framework::Tensor>();
    float momentum = ctx.op_.GetAttr<float>("momentum");
    bool nesterov = ctx.op_.GetAttr<int>("nesterov");

    momentum_sgd_update(size,
                        grad.data<float>(),
                        moment.data<float>(),
                        output_grad.raw_data<float>(),
                        output_moment.raw_data<float>(),
                        learning_rate.raw_data<float>(),
                        momentum,
                        nesterov,
                        output_param.raw_data<float>());
  }
};

}  // namespace operators
}  // namespace paddle
