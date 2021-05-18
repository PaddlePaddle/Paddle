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

#include "paddle/fluid/operators/reduce_ops/reduce_functor_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cuh"
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_prod_op.h"

namespace paddle {
namespace operators {

template <typename T>
class ReduceProdKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    bool reduce_all = context.Attr<bool>("reduce_all");
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");
    auto out_dtype = context.Attr<int>("out_dtype");

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
    if (out_dtype >= 0) {
#define VisitDataTypeSmall_t(cpp_type, proto_type)                             \
  do {                                                                         \
    if (static_cast<framework::proto::VarType::Type>(out_dtype) ==             \
        proto_type) {                                                          \
      TensorReduce<T, cpp_type, CustomMul<cpp_type>,                           \
                   detail::IdentityFunctor<cpp_type>>(                         \
          *input, output, reduce_dims, static_cast<cpp_type>(1.0f),            \
          CustomMul<cpp_type>(), detail::IdentityFunctor<cpp_type>(), stream); \
    }                                                                          \
  } while (0)
      _ForEachDataTypeSmall_(VisitDataTypeSmall_t);
#undef VisitDataTypeSmall_t
    } else {
      TensorReduce<T, T, CustomMul<T>, detail::IdentityFunctor<T>>(
          *input, output, reduce_dims, static_cast<T>(1.0f), CustomMul<T>(),
          detail::IdentityFunctor<T>(), stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

#ifdef __HIPCC__
// Eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:922
// do not support double in HIPCC platform (Eigen3 to be fixed)
REGISTER_OP_CUDA_KERNEL(reduce_prod, ops::ReduceProdKernel<float>,
                        ops::ReduceProdKernel<int>,
                        ops::ReduceProdKernel<int64_t>);
#else
REGISTER_OP_CUDA_KERNEL(reduce_prod, ops::ReduceProdKernel<float>,
                        ops::ReduceProdKernel<int>,
                        ops::ReduceProdKernel<double>,
                        ops::ReduceProdKernel<int64_t>);
#endif
