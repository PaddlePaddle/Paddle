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

#include "paddle/fluid/operators/kernel_primitives/compute_primitives.h"
#include "paddle/pten/api/ext/dispatch.h"
#include "paddle/pten/api/include/tensor.h"
#include "paddle/pten/kernels/cuda/utils.h"
#include "paddle/pten/kernels/hybird/cuda/reduce/reduce_cuda_helper.h"
#include "paddle/pten/kernels/hybird/math/cast_func.h"

namespace pten {
namespace detail {

static void AsyncCopy(const DenseTensor& src, DenseTensor* dst) {
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  const paddle::platform::CUDADeviceContext* dev_ctx;
  if (paddle::platform::is_gpu_place(dst->place()) ||
      paddle::platform::is_npu_place(dst->place())) {
    dev_ctx = static_cast<paddle::platform::CUDADeviceContext*>(
        pool.Get(dst->place()));

  } else {
    dev_ctx = static_cast<paddle::platform::CUDADeviceContext*>(
        pool.Get(src.place()));
  }

  pten::Copy(*dev_ctx, src, false, dst);
}

template <typename Tx,
          typename Ty,
          template <typename, typename> class ReduceOp>
void TensorReduceFunctorImpl(const pten::DenseTensor& x,
                             pten::DenseTensor* y,
                             std::vector<int64_t> origin_reduce_dims,
                             gpuStream_t stream) {
  // Allocate memory
  y->mutable_data<Ty>();
  auto x_dim = paddle::framework::vectorize<int64_t>(x.dims());
  auto config = ReduceConfig<Ty>(origin_reduce_dims, x_dim);
  config.Run();
  int64_t numel = x.numel();
  // after config.run()
  // SetOutputData for ReduceHigherDim when should_reduce_again is true,
  // temp_output should be stored temp_data in output_data space or stored in
  // y_data;
  pten::DDim tmp_ddim;
  pten::DenseTensor tmp = pten::DenseTensor(
      pten::make_intrusive<paddle::experimental::SharedStorage>(y->place()),
      pten::DenseTensorMeta(y->dtype(), tmp_ddim, y->layout()));

  auto x_data = x.data<Tx>();
  auto y_data = y->mutable_data<Ty>();

  auto* dev_ctx = static_cast<paddle::platform::CUDADeviceContext*>(
      paddle::platform::DeviceContextPool::Instance().Get(x.place()));
  if (config.reduce_num == 1) {
    auto out_dims = y->dims();
    if (x.dtype() == y->dtype()) {
      AsyncCopy(x, y);
      y->Resize(out_dims);
    } else {
      PD_VISIT_ALL_TYPES(y->dtype(), "CastKernelImpl", ([&] {
                           pten::math::CastKernelImpl<CUDAContext, Tx, data_t>(
                               *dev_ctx, x, y);
                         }));
    }
    return;
  }

  config.SetOutputData(y_data, x.place(), &tmp);
  bool use_cub_reduce = (config.reduce_num == numel) &&
                        (!std::is_same<Tx, paddle::platform::float16>::value);
  if (use_cub_reduce) {
    // launch CUB::Reduce
    using TransformOp = typename ReduceOp<Tx, Ty>::Transformer;
    auto reducer = ReduceOp<Tx, Ty>();
    cub::TransformInputIterator<Ty, TransformOp, const Tx*> trans_x(
        x_data, TransformOp(config.reduce_num));
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce(nullptr,
                              temp_storage_bytes,
                              trans_x,
                              y_data,
                              config.reduce_num,
                              reducer,
                              reducer.initial(),
                              stream);
    // framework::Tensor tmp;
    pten::DenseTensor tmp = pten::DenseTensor(
        pten::make_intrusive<paddle::experimental::SharedStorage>(x.place()),
        pten::DenseTensorMeta(pten::DataType::UINT8,
                              paddle::framework::make_ddim(
                                  {static_cast<int64_t>(temp_storage_bytes)}),
                              x.layout()));
    auto* temp_storage = tmp.mutable_data<uint8_t>();
    cub::DeviceReduce::Reduce(temp_storage,
                              temp_storage_bytes,
                              trans_x,
                              y_data,
                              config.reduce_num,
                              reducer,
                              reducer.initial(),
                              stream);

    return;
  }

  using MPType =
      typename paddle::operators::kernel_primitives::details::MPTypeTrait<
          Ty>::Type;
  auto reducer = ReduceOp<Tx, MPType>();
  // launch ReduceHigherDimKernel
  // when reduce_dim.size() == 1 and reduce_dim[0] != x_dim.size() - 1, this
  // function will be used
  // eg: x_dim = {nz, ny, nx}, nx != 1, axis can be 0 or 1
  //     if axis = 1 then grid.z = nz, grid.y = ny / block_size, grid.x = nx /
  //     32
  //     else grid.z = 1, grid.y = ny / block_size, grid.x = nx /32
  if (config.reduce_type == ReduceType::kReduceHigherDim) {
    using TransformOp = typename ReduceOp<Tx, MPType>::Transformer;
    kps::DimConfig dim = kps::DimConfig(config.grid.x,
                                        config.grid.y,
                                        config.grid.z,
                                        config.block.x,
                                        config.blocking_size,
                                        0);
    dim.SetRem(config.left_num % config.block.x,
               config.reduce_num % config.blocking_size,
               0);

#ifdef PADDLE_WITH_XPU2
    pten::detail::ReduceHigherDimKernel<Tx,
                                        Ty,
                                        MPType,
                                        ReduceOp<Tx, MPType>,
                                        TransformOp><<<8, 128, stream>>>(
        x_data,
        config.output_data,
        reducer,
        TransformOp(config.reduce_num),
        reducer.initial(),
        config.reduce_num,
        config.left_num,
        config.blocking_size,
        dim);
#else
    pten::detail::ReduceHigherDimKernel<
        Tx,
        Ty,
        MPType,
        ReduceOp<Tx, MPType>,
        TransformOp><<<config.grid, config.block, 0, stream>>>(
        x_data,
        config.output_data,
        reducer,
        TransformOp(config.reduce_num),
        reducer.initial(),
        config.reduce_num,
        config.left_num,
        config.blocking_size,
        dim);
#endif

    if (config.should_reduce_again) {
      dim3 block = dim3(config.block.x, 1, 1);
      dim3 grid = dim3(config.grid.x, 1, config.grid.z);
      kps::DimConfig dim2 =
          kps::DimConfig(grid.x, grid.y, grid.z, block.x, config.grid.y, 0);
      dim2.SetRem(config.left_num % config.block.x, 0, 0);

#ifdef PADDLE_WITH_XPU2
      pten::detail::ReduceHigherDimKernel<
          Ty,
          Ty,
          MPType,
          ReduceOp<Tx, MPType>,
          kps::IdentityFunctor<Ty, MPType>><<<8, 128, stream>>>(
          config.output_data,
          y_data,
          reducer,
          kps::IdentityFunctor<Ty, MPType>(config.grid.y),
          reducer.initial(),
          config.grid.y,
          config.left_num,
          config.grid.y,
          dim2);
#else
      pten::detail::ReduceHigherDimKernel<
          Ty,
          Ty,
          MPType,
          ReduceOp<Tx, MPType>,
          kps::IdentityFunctor<Ty, MPType>><<<grid, block, 0, stream>>>(
          config.output_data,
          y_data,
          reducer,
          kps::IdentityFunctor<Ty, MPType>(config.grid.y),
          reducer.initial(),
          config.grid.y,
          config.left_num,
          config.grid.y,
          dim2);
#endif
    }
    return;
  }

  // when reduce_dim.size() == 1 and reduce_dim[0] == x_dim.size() - 1, or
  // when reduce_dim.size() != 1 and reduce_dim.size() != x_dim.size(), this
  // function will be used
  LaunchReduceKernel<Tx, Ty, MPType, ReduceOp<Tx, MPType>>(
      x_data, y_data, reducer, reducer.initial(), stream, config);
}

}  // namespace detail
}  // namespace pten
