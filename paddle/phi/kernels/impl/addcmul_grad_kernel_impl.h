// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/common/ddim.h"
#include "paddle/common/errors.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

// Helper: compute reshape_dims and reduce_dims for broadcast gradient
inline void ComputeBroadcastGradDims(const std::vector<int64_t>& grad_dims,
                                     const std::vector<int64_t>& out_dims,
                                     std::vector<int>* reshape_dims_vec,
                                     std::vector<int>* reduce_dims_vec) {
  std::vector<int64_t> extended = grad_dims;
  size_t diff = out_dims.size() - grad_dims.size();
  extended.insert(extended.begin(), diff, 1);

  reshape_dims_vec->clear();
  reduce_dims_vec->clear();
  for (size_t i = 0; i < extended.size(); ++i) {
    reduce_dims_vec->push_back(static_cast<int>(reshape_dims_vec->size()));
    reshape_dims_vec->push_back(static_cast<int>(out_dims[i] / extended[i]));
    reshape_dims_vec->push_back(static_cast<int>(extended[i]));
  }
}

// Use EigenBroadcastGrad for reduction (robust on GPU)
template <typename Context, typename T, int Dims>
static void ReduceGrad(const Context& dev_ctx,
                       const DenseTensor& src,
                       const std::vector<int>& reshape_dims_vec,
                       const std::vector<int>& reduce_dims_vec,
                       DenseTensor* dst) {
  dev_ctx.template Alloc<T>(dst);
  Eigen::DSizes<Eigen::DenseIndex, Dims * 2> reshape_dims;
  Eigen::DSizes<Eigen::DenseIndex, Dims> reduce_dims;
  for (size_t i = 0; i < reshape_dims_vec.size(); ++i) {
    reshape_dims[i] = reshape_dims_vec[i];
  }
  for (size_t i = 0; i < reduce_dims_vec.size(); ++i) {
    reduce_dims[i] = reduce_dims_vec[i];
  }
  auto src_flat = EigenVector<T>::Flatten(src);
  auto dst_flat = EigenVector<T>::Flatten(*dst);
  auto& place = *dev_ctx.eigen_device();
  funcs::EigenBroadcastGrad<std::decay_t<decltype(place)>, T, Dims>::Eval(
      place, dst_flat, src_flat, reduce_dims, reshape_dims);
}

template <typename T, typename Context, int Dims>
static void AddcmulGradImpl(const Context& dev_ctx,
                            const DenseTensor& tensor1,
                            const DenseTensor& tensor2,
                            const DenseTensor& out_grad,
                            const Scalar& value,
                            DenseTensor* input_grad,
                            DenseTensor* tensor1_grad,
                            DenseTensor* tensor2_grad) {
  using MPType = typename dtype::MPTypeTrait<T>::Type;
  auto& place = *dev_ctx.eigen_device();
  MPType val = static_cast<MPType>(value.to<float>());
  auto out_dims = common::vectorize<int64_t>(out_grad.dims());

  // Extend dims helper
  auto extend_dims = [&](const DDim& dims) {
    auto v = common::vectorize<int64_t>(dims);
    size_t diff = out_dims.size() - v.size();
    v.insert(v.begin(), diff, 1);
    return v;
  };

  auto compute_bcast = [&](const std::vector<int64_t>& ext) {
    Eigen::DSizes<Eigen::DenseIndex, Dims> bcast;
    for (size_t i = 0; i < out_dims.size(); ++i) {
      bcast[i] = out_dims[i] / ext[i];
    }
    return bcast;
  };

  // d(input) = reduce_sum(dout)
  if (input_grad) {
    if (input_grad->dims() == out_grad.dims()) {
      dev_ctx.template Alloc<T>(input_grad);
      phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, input_grad);
    } else {
      std::vector<int> reshape_vec, reduce_vec;
      ComputeBroadcastGradDims(common::vectorize<int64_t>(input_grad->dims()),
                               out_dims,
                               &reshape_vec,
                               &reduce_vec);
      ReduceGrad<Context, T, Dims>(
          dev_ctx, out_grad, reshape_vec, reduce_vec, input_grad);
    }
  }

  // d(tensor1) = reduce_sum(dout * value * tensor2)
  if (tensor1_grad) {
    auto ext_t2 = extend_dims(tensor2.dims());
    auto t2_bcast = compute_bcast(ext_t2);
    auto eigen_t2 =
        EigenTensor<T, Dims>::From(tensor2, common::make_ddim(ext_t2));
    auto eigen_dout = EigenTensor<T, Dims>::From(out_grad);

    if (tensor1_grad->dims() == out_grad.dims()) {
      // No reduction needed - compute directly into tensor1_grad
      dev_ctx.template Alloc<T>(tensor1_grad);
      auto eigen_tensor1_grad = EigenTensor<T, Dims>::From(*tensor1_grad);
      eigen_tensor1_grad.device(place) =
          (eigen_dout.template cast<MPType>() * val *
           eigen_t2.broadcast(t2_bcast).template cast<MPType>())
              .template cast<T>();
    } else {
      // Reduction needed - allocate intermediate tensor for broadcast result
      DenseTensor tensor1_grad_broadcast;
      tensor1_grad_broadcast.Resize(out_grad.dims());
      dev_ctx.template Alloc<T>(&tensor1_grad_broadcast);
      auto eigen_tensor1_grad_broadcast =
          EigenTensor<T, Dims>::From(tensor1_grad_broadcast);
      eigen_tensor1_grad_broadcast.device(place) =
          (eigen_dout.template cast<MPType>() * val *
           eigen_t2.broadcast(t2_bcast).template cast<MPType>())
              .template cast<T>();

      std::vector<int> reshape_vec, reduce_vec;
      ComputeBroadcastGradDims(common::vectorize<int64_t>(tensor1_grad->dims()),
                               out_dims,
                               &reshape_vec,
                               &reduce_vec);
      ReduceGrad<Context, T, Dims>(dev_ctx,
                                   tensor1_grad_broadcast,
                                   reshape_vec,
                                   reduce_vec,
                                   tensor1_grad);
    }
  }

  // d(tensor2) = reduce_sum(dout * value * tensor1)
  if (tensor2_grad) {
    auto ext_t1 = extend_dims(tensor1.dims());
    auto t1_bcast = compute_bcast(ext_t1);
    auto eigen_t1 =
        EigenTensor<T, Dims>::From(tensor1, common::make_ddim(ext_t1));
    auto eigen_dout = EigenTensor<T, Dims>::From(out_grad);

    if (tensor2_grad->dims() == out_grad.dims()) {
      // No reduction needed - compute directly into tensor2_grad
      dev_ctx.template Alloc<T>(tensor2_grad);
      auto eigen_tensor2_grad = EigenTensor<T, Dims>::From(*tensor2_grad);
      eigen_tensor2_grad.device(place) =
          (eigen_dout.template cast<MPType>() * val *
           eigen_t1.broadcast(t1_bcast).template cast<MPType>())
              .template cast<T>();
    } else {
      // Reduction needed - allocate intermediate tensor for broadcast result
      DenseTensor tensor2_grad_broadcast;
      tensor2_grad_broadcast.Resize(out_grad.dims());
      dev_ctx.template Alloc<T>(&tensor2_grad_broadcast);
      auto eigen_tensor2_grad_broadcast =
          EigenTensor<T, Dims>::From(tensor2_grad_broadcast);
      eigen_tensor2_grad_broadcast.device(place) =
          (eigen_dout.template cast<MPType>() * val *
           eigen_t1.broadcast(t1_bcast).template cast<MPType>())
              .template cast<T>();

      std::vector<int> reshape_vec, reduce_vec;
      ComputeBroadcastGradDims(common::vectorize<int64_t>(tensor2_grad->dims()),
                               out_dims,
                               &reshape_vec,
                               &reduce_vec);
      ReduceGrad<Context, T, Dims>(dev_ctx,
                                   tensor2_grad_broadcast,
                                   reshape_vec,
                                   reduce_vec,
                                   tensor2_grad);
    }
  }
}

template <typename Context, typename T>
static void AddcmulGradZero(const Context& dev_ctx,
                            const DenseTensor& tensor1,
                            const DenseTensor& tensor2,
                            const DenseTensor& out_grad,
                            const Scalar& value,
                            DenseTensor* input_grad,
                            DenseTensor* t1_grad,
                            DenseTensor* t2_grad) {
  auto dim = ::common::make_ddim(std::vector<int64_t>(1, 1));
  using MPType = typename dtype::MPTypeTrait<T>::Type;
  auto& place = *dev_ctx.eigen_device();
  MPType val = static_cast<MPType>(value.to<float>());

  auto eigen_t1 = phi::EigenTensor<T, 1>::From(tensor1, dim);
  auto eigen_t2 = phi::EigenTensor<T, 1>::From(tensor2, dim);
  auto eigen_dout = phi::EigenTensor<T, 1>::From(out_grad, dim);

  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    auto eigen_dx = phi::EigenTensor<T, 1>::From(*input_grad, dim);
    eigen_dx.device(place) = eigen_dout;
  }
  if (t1_grad) {
    dev_ctx.template Alloc<T>(t1_grad);
    auto eigen_dt1 = phi::EigenTensor<T, 1>::From(*t1_grad, dim);
    eigen_dt1.device(place) = (eigen_dout.template cast<MPType>() *
                               eigen_t2.template cast<MPType>() * val)
                                  .template cast<T>();
  }
  if (t2_grad) {
    dev_ctx.template Alloc<T>(t2_grad);
    auto eigen_dt2 = phi::EigenTensor<T, 1>::From(*t2_grad, dim);
    eigen_dt2.device(place) = (eigen_dout.template cast<MPType>() *
                               eigen_t1.template cast<MPType>() * val)
                                  .template cast<T>();
  }
}

template <typename T, typename Context>
void AddcmulGradKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const DenseTensor& tensor1,
                       const DenseTensor& tensor2,
                       const DenseTensor& out_grad,
                       const Scalar& value,
                       DenseTensor* input_grad,
                       DenseTensor* tensor1_grad,
                       DenseTensor* tensor2_grad) {
  if (out_grad.numel() == 0) {
    if (input_grad)
      phi::Full<T, Context>(
          dev_ctx,
          phi::IntArray(::common::vectorize(input_grad->dims())),
          0,
          input_grad);
    if (tensor1_grad)
      phi::Full<T, Context>(
          dev_ctx,
          phi::IntArray(::common::vectorize(tensor1_grad->dims())),
          0,
          tensor1_grad);
    if (tensor2_grad)
      phi::Full<T, Context>(
          dev_ctx,
          phi::IntArray(::common::vectorize(tensor2_grad->dims())),
          0,
          tensor2_grad);
    return;
  }

  int rank = out_grad.dims().size();
  switch (rank) {
    case 0:
      AddcmulGradZero<Context, T>(dev_ctx,
                                  tensor1,
                                  tensor2,
                                  out_grad,
                                  value,
                                  input_grad,
                                  tensor1_grad,
                                  tensor2_grad);
      break;
    case 1:
      AddcmulGradImpl<T, Context, 1>(dev_ctx,
                                     tensor1,
                                     tensor2,
                                     out_grad,
                                     value,
                                     input_grad,
                                     tensor1_grad,
                                     tensor2_grad);
      break;
    case 2:
      AddcmulGradImpl<T, Context, 2>(dev_ctx,
                                     tensor1,
                                     tensor2,
                                     out_grad,
                                     value,
                                     input_grad,
                                     tensor1_grad,
                                     tensor2_grad);
      break;
    case 3:
      AddcmulGradImpl<T, Context, 3>(dev_ctx,
                                     tensor1,
                                     tensor2,
                                     out_grad,
                                     value,
                                     input_grad,
                                     tensor1_grad,
                                     tensor2_grad);
      break;
    case 4:
      AddcmulGradImpl<T, Context, 4>(dev_ctx,
                                     tensor1,
                                     tensor2,
                                     out_grad,
                                     value,
                                     input_grad,
                                     tensor1_grad,
                                     tensor2_grad);
      break;
    case 5:
      AddcmulGradImpl<T, Context, 5>(dev_ctx,
                                     tensor1,
                                     tensor2,
                                     out_grad,
                                     value,
                                     input_grad,
                                     tensor1_grad,
                                     tensor2_grad);
      break;
    case 6:
      AddcmulGradImpl<T, Context, 6>(dev_ctx,
                                     tensor1,
                                     tensor2,
                                     out_grad,
                                     value,
                                     input_grad,
                                     tensor1_grad,
                                     tensor2_grad);
      break;
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "AddcmulGrad only supports rank <= 6, got %d", rank));
  }
}

}  // namespace phi
