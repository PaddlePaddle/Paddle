/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/where_index_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"
#include "paddle/phi/kernels/select_impl.cu.h"

namespace paddle {
namespace operators {
template <typename T1, typename T2, typename OutT>
struct IndexFunctor {
  T2 stride[paddle::framework::DDim::kMaxRank];
  int dims;
  explicit IndexFunctor(const framework::DDim &in_dims) {
    dims = in_dims.size();
    std::vector<T2> strides_in_tmp;
    strides_in_tmp.resize(dims, 1);
    // get strides according to in_dims
    for (T2 i = 1; i < dims; i++) {
      strides_in_tmp[i] = strides_in_tmp[i - 1] * in_dims[dims - i];
    }
    memcpy(stride, strides_in_tmp.data(), dims * sizeof(T2));
  }

  HOSTDEVICE inline void operator()(OutT *out, const T1 *mask, const T2 *index,
                                    const int num) {
    int store_fix = 0;
    for (int idx = 0; idx < num; idx++) {
      if (mask[idx]) {
        T2 data_index = index[idx];
        // get index
        for (int rank_id = dims - 1; rank_id >= 0; --rank_id) {
          out[store_fix] = static_cast<OutT>(data_index / stride[rank_id]);
          data_index = data_index % stride[rank_id];
          store_fix++;
        }
      }
    }
  }
};

template <typename T>
class CUDAWhereIndexKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *condition = context.Input<framework::Tensor>("Condition");
    auto *out = context.Output<framework::Tensor>("Out");
    using CUDADeviceContext = paddle::platform::CUDADeviceContext;
    auto &dev_ctx = context.template device_context<CUDADeviceContext>();
    framework::Tensor in_data;
    auto dims = condition->dims();
    using Functor = IndexFunctor<T, int64_t, int64_t>;
    Functor index_functor = Functor(dims);
    phi::SelectKernel<T, T, int64_t, 0, Functor>(
        static_cast<const phi::GPUContext &>(dev_ctx), *condition, in_data, out,
        index_functor);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(where_index, ops::CUDAWhereIndexKernel<int64_t>,
                        ops::CUDAWhereIndexKernel<int>,
                        ops::CUDAWhereIndexKernel<int16_t>,
                        ops::CUDAWhereIndexKernel<bool>,
                        ops::CUDAWhereIndexKernel<float>,
                        ops::CUDAWhereIndexKernel<double>);
