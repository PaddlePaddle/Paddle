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

#include <vector>
#include "paddle/fluid/operators/cub_reduce.h"
#include "paddle/fluid/operators/reduce_mean_op.h"

namespace paddle {
namespace operators {

struct DivideFunctor {
  HOSTDEVICE explicit inline DivideFunctor(int n) : n_(n) {}

  template <typename T>
  HOSTDEVICE inline T operator()(const T& x) const {
    return x / n_;
  }

 private:
  int n_;
};

template <typename T>
class ReduceMeanKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    bool reduce_all = context.Attr<bool>("reduce_all");
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");

    auto dims = context.Attr<std::vector<int>>("dim");
    bool keep_dim = context.Attr<bool>("keep_dim");

    std::vector<int> reduce_dims;
    if (reduce_all) {
      reduce_dims.resize(input->dims().size());
      for (int i = 0; i < reduce_dims.size(); ++i) reduce_dims[i] = i;
    } else {
      for (auto e : dims) {
        reduce_dims.push_back(e >= 0 ? e : e + input->dims().size());
      }
    }

    int reduce_num = 1;
    for (int i = 0; i < reduce_dims.size(); ++i) {
      reduce_num *= input->dims()[reduce_dims[i]];
    }

    auto stream = context.cuda_device_context().stream();
    TensorReduce<T, T, cub::Sum, DivideFunctor>(
        *input, output, reduce_dims, static_cast<T>(0), cub::Sum(),
        DivideFunctor(reduce_num), stream);
    if (keep_dim) {
      framework::DDim out_dim = input->dims();
      for (int i = 0; i < reduce_dims.size(); ++i) out_dim[reduce_dims[i]] = 1;
      output->Resize(out_dim);
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(reduce_mean,
                        ops::ReduceKernel<paddle::platform::CUDADeviceContext,
                                          float, ops::MeanFunctor>,
                        ops::ReduceKernel<paddle::platform::CUDADeviceContext,
                                          double, ops::MeanFunctor>,
                        ops::ReduceKernel<paddle::platform::CUDADeviceContext,
                                          int, ops::MeanFunctor>,
                        ops::ReduceKernel<paddle::platform::CUDADeviceContext,
                                          int64_t, ops::MeanFunctor>);
REGISTER_OP_CUDA_KERNEL(
    reduce_mean_grad, ops::ReduceGradKernel<paddle::platform::CUDADeviceContext,
                                            float, ops::MeanGradFunctor>,
    ops::ReduceGradKernel<paddle::platform::CUDADeviceContext, double,
                          ops::MeanGradFunctor>,
    ops::ReduceGradKernel<paddle::platform::CUDADeviceContext, int,
                          ops::MeanGradFunctor>,
    ops::ReduceGradKernel<paddle::platform::CUDADeviceContext, int64_t,
                          ops::MeanGradFunctor>);
