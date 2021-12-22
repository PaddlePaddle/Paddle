// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif

#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/fluid/framework/array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/cast_op.h"
#include "paddle/fluid/operators/kernel_primitives/kernel_primitives.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/fast_divmod.h"
#include "paddle/fluid/string/string_helper.h"

#include "paddle/pten/api/ext/dispatch.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/gpu/utils.h"
#include "paddle/pten/kernels/hybird/cuda/reduce/reduce_cuda_impl.h"
#include "paddle/pten/kernels/hybird/math/cast_func.h"

namespace paddle {
namespace operators {

template <typename Tx, typename Ty, template <typename> class ReduceOp,
          typename TransformOp>
void TensorReduceFunctorImpl(const framework::Tensor& x, framework::Tensor* y,
                             const TransformOp& transform,
                             const std::vector<int>& origin_reduce_dims,
                             gpuStream_t stream) {
  auto pt_x = paddle::experimental::MakePtenDenseTensor(x);
  auto pt_y = paddle::experimental::MakePtenDenseTensor(*y);

  pten::kernels::TensorReduceFunctorImpl<Tx, Ty, ReduceOp, TransformOp>(
      *pt_x.get(), pt_y.get(), transform, origin_reduce_dims, stream);

  /*
    auto x_dim = framework::vectorize<int>(x.dims());
    auto config = ReduceConfig<Ty>(origin_reduce_dims, x_dim);
    config.Run();
    int numel = x.numel();
    // after config.run()
    // SetOutputData for ReduceHigherDim when should_reduce_again is true,
    // temp_output should be stored temp_data in output_data space or stored in
    // y_data;
    framework::Tensor tmp;
    auto x_data = x.data<Tx>();
    auto y_data = y->mutable_data<Ty>(x.place());

    if (config.reduce_num == 1) {
      auto out_dims = y->dims();
      if (x.type() == y->type()) {
        framework::TensorCopy(x, y->place(), y);
        y->Resize(out_dims);
      } else {
        auto* dev_ctx = static_cast<platform::CUDADeviceContext*>(
            paddle::platform::DeviceContextPool::Instance().Get(x.place()));
        framework::VisitDataType(
            static_cast<framework::proto::VarType::Type>(y->type()),
            CastOpFunctor<platform::CUDADeviceContext, Tx>(&x, y, *dev_ctx));
      }
      return;
    }

    config.SetOutputData(y_data, x.place(), &tmp);
    constexpr bool kIsTxFP16 = std::is_same<Tx,
  paddle::platform::float16>::value;
    bool use_cub_reduce = config.reduce_num == numel && !kIsTxFP16;
    if (use_cub_reduce) {
      CubTensorReduceFunctorImpl<Tx, Ty, ReduceOp, TransformOp>(
          x_data, y_data, transform, config.reduce_num, x.place(), stream);
      return;
    }

    using MPType = typename details::MPTypeTrait<Ty>::Type;
    auto reducer = ReduceOp<MPType>();
    // launch ReduceHigherDimKernel
    // when reduce_dim.size() == 1 and reduce_dim[0] != x_dim.size() - 1, this
    // function will be used
    // eg: x_dim = {nz, ny, nx}, nx != 1, axis can be 0 or 1
    //     if axis = 1 then grid.z = nz, grid.y = ny / block_size, grid.x = nx /
    //     32
    //     else grid.z = 1, grid.y = ny / block_size, grid.x = nx /32
    if (config.reduce_type == ReduceType::kReduceHigherDim) {
      kps::DimConfig dim =
          kps::DimConfig(config.grid.x, config.grid.y, config.grid.z,
                         config.block.x, config.blocking_size, 0);
      dim.SetRem(config.left_num % config.block.x,
                 config.reduce_num % config.blocking_size, 0);

  #ifdef PADDLE_WITH_XPU2
      ReduceHigherDimKernel<Tx, Ty, MPType, ReduceOp<MPType>,
                            TransformOp><<<8, 128, stream>>>(
          x_data, config.output_data, reducer, transform, reducer.initial(),
          config.reduce_num, config.left_num, config.blocking_size, dim);
  #else
      ReduceHigherDimKernel<
          Tx, Ty, MPType, ReduceOp<MPType>,
          TransformOp><<<config.grid, config.block, 0, stream>>>(
          x_data, config.output_data, reducer, transform, reducer.initial(),
          config.reduce_num, config.left_num, config.blocking_size, dim);
  #endif

      if (config.should_reduce_again) {
        dim3 block = dim3(config.block.x, 1, 1);
        dim3 grid = dim3(config.grid.x, 1, config.grid.z);
        kps::DimConfig dim2 =
            kps::DimConfig(grid.x, grid.y, grid.z, block.x, config.grid.y, 0);
        dim2.SetRem(config.left_num % config.block.x, 0, 0);

  #ifdef PADDLE_WITH_XPU2
        ReduceHigherDimKernel<
            Ty, Ty, MPType, ReduceOp<MPType>,
            kps::IdentityFunctor<Ty, MPType>><<<8, 128, stream>>>(
            config.output_data, y_data, reducer,
            kps::IdentityFunctor<Ty, MPType>(config.grid.y), reducer.initial(),
            config.grid.y, config.left_num, config.grid.y, dim2);
  #else
        ReduceHigherDimKernel<
            Ty, Ty, MPType, ReduceOp<MPType>,
            kps::IdentityFunctor<Ty, MPType>><<<grid, block, 0, stream>>>(
            config.output_data, y_data, reducer,
            kps::IdentityFunctor<Ty, MPType>(config.grid.y), reducer.initial(),
            config.grid.y, config.left_num, config.grid.y, dim2);
  #endif
      }
      return;
    }

    // when reduce_dim.size() == 1 and reduce_dim[0] == x_dim.size() - 1, or
    // when reduce_dim.size() != 1 and reduce_dim.size() != x_dim.size(), this
    // function will be used
    LaunchReduceKernel<Tx, Ty, MPType, ReduceOp<MPType>, TransformOp>(
        x_data, y_data, reducer, transform, reducer.initial(), stream, config);
  */
}

}  // namespace operators
}  // namespace paddle
