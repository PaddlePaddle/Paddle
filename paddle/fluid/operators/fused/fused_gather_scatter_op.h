/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
struct FusedGatherScatterSumFunctor {
  void operator()(const int& first_flag, const Tensor& src_slice,
                  Tensor* dst_slice) {
    auto eigen_src = framework::EigenVector<T>::Flatten(src_slice);
    auto eigen_dst = framework::EigenVector<T>::Flatten(*dst_slice);
    eigen_dst += eigen_src;
  }
};

template <typename T>
struct FusedGatherScatterMinFunctor {
  void operator()(const int& first_flag, const Tensor& src_slice,
                  Tensor* dst_slice) {
    auto eigen_src = framework::EigenVector<T>::Flatten(src_slice);
    auto eigen_dst = framework::EigenVector<T>::Flatten(*dst_slice);
    if (first_flag == 0) {
      eigen_dst += eigen_src;
    } else {
      eigen_dst = eigen_dst.cwiseMin(eigen_src);
    }
  }
};

template <typename T>
struct FusedGatherScatterMaxFunctor {
  void operator()(const int& first_flag, const Tensor& src_slice,
                  Tensor* dst_slice) {
    auto eigen_src = framework::EigenVector<T>::Flatten(src_slice);
    auto eigen_dst = framework::EigenVector<T>::Flatten(*dst_slice);
    if (first_flag == 0) {
      eigen_dst += eigen_src;
    } else {
      eigen_dst = eigen_dst.cwiseMax(eigen_src);
    }
  }
};

template <typename T, typename IndexT, typename Functor>
void elementwise_inner_operation(const Tensor& src, Tensor* dst,
                                 const IndexT& src_index,
                                 const IndexT& dst_index,
                                 const bool& first_flag, Functor functor) {
  auto src_slice = src.Slice(src_index, src_index + 1);
  auto dst_slice = dst->Slice(dst_index, dst_index + 1);

  functor(first_flag, src_slice, &dst_slice);
}

template <typename T, typename IndexT, typename Functor>
void gather_scatter_cpu_for_loop(const int index_size, const IndexT* g_index,
                                 const IndexT* s_index, const Tensor& src,
                                 Tensor* dst, const std::string& pool_type) {
  Functor functor;
  if (pool_type == "MIN" || pool_type == "MAX") {
    std::set<IndexT> existed_dst;
    for (int i = 0; i < index_size; ++i) {
      IndexT src_ptr = g_index[i];
      IndexT dst_ptr = s_index[i];
      int nRet = std::count(existed_dst.begin(), existed_dst.end(), dst_ptr);
      if (nRet == 0) {
        elementwise_inner_operation<T, IndexT, Functor>(src, dst, src_ptr,
                                                        dst_ptr, 0, functor);
        existed_dst.insert(dst_ptr);
      } else {
        elementwise_inner_operation<T, IndexT, Functor>(src, dst, src_ptr,
                                                        dst_ptr, 1, functor);
      }
    }
  } else if (pool_type == "SUM" || pool_type == "MEAN") {
    for (int i = 0; i < index_size; ++i) {
      IndexT src_ptr = g_index[i];
      IndexT dst_ptr = s_index[i];
      elementwise_inner_operation<T, IndexT, Functor>(src, dst, src_ptr,
                                                      dst_ptr, 0, functor);
    }
  }
}

template <typename DeviceContext, typename T, typename IndexT>
class FusedGatherScatterOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<Tensor>("X");
    auto* gather_index = ctx.Input<Tensor>("Gather_index");
    auto* scatter_index = ctx.Input<Tensor>("Scatter_index");
    auto* Y = ctx.Output<Tensor>("Out");

    int index_size = gather_index->dims()[0];

    T* p_output = Y->mutable_data<T>(ctx.GetPlace());
    auto src_dims = X->dims();
    int64_t memset_size = 1;
    for (int i = 0; i < src_dims.size(); ++i) memset_size *= src_dims[i];
    const size_t& memset_bytes = memset_size * sizeof(T);
    memset(p_output, 0, memset_bytes);

    const IndexT* g_index = gather_index->data<IndexT>();
    const IndexT* s_index = scatter_index->data<IndexT>();

    std::string pool_type = ctx.Attr<std::string>("pool_type");
    if (pool_type == "SUM") {
      gather_scatter_cpu_for_loop<T, IndexT, FusedGatherScatterSumFunctor<T>>(
          index_size, g_index, s_index, *X, Y, pool_type);
    } else if (pool_type == "MIN") {
      gather_scatter_cpu_for_loop<T, IndexT, FusedGatherScatterMinFunctor<T>>(
          index_size, g_index, s_index, *X, Y, pool_type);
    } else if (pool_type == "MAX") {
      gather_scatter_cpu_for_loop<T, IndexT, FusedGatherScatterMaxFunctor<T>>(
          index_size, g_index, s_index, *X, Y, pool_type);
    } else if (pool_type == "MEAN") {
      gather_scatter_cpu_for_loop<T, IndexT, FusedGatherScatterSumFunctor<T>>(
          index_size, g_index, s_index, *X, Y, pool_type);
      // TODO(daisiming): Add mean operation.
    }
  }
};

template <typename DeviceContext, typename T, typename IndexT>
class FusedGatherScatterGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* gather_index = ctx.Input<Tensor>("Gather_index");
    auto* scatter_index = ctx.Input<Tensor>("Scatter_index");
    auto* Y = ctx.Output<Tensor>(framework::GradVarName("X"));

    int index_size = gather_index->dims()[0];

    T* p_output = Y->mutable_data<T>(ctx.GetPlace());
    auto src_dims = X->dims();
    int64_t memset_size = 1;
    for (int i = 0; i < src_dims.size(); ++i) memset_size *= src_dims[i];
    const size_t& memset_bytes = memset_size * sizeof(T);
    memset(p_output, 0, memset_bytes);

    const IndexT* g_index = gather_index->data<IndexT>();
    const IndexT* s_index = scatter_index->data<IndexT>();

    std::string pool_type = ctx.Attr<std::string>("pool_type");
    if (pool_type == "SUM") {
      gather_scatter_cpu_for_loop<T, IndexT, FusedGatherScatterSumFunctor<T>>(
          index_size, g_index, s_index, *X, Y, pool_type);
    } else if (pool_type == "MIN") {
      gather_scatter_cpu_for_loop<T, IndexT, FusedGatherScatterMinFunctor<T>>(
          index_size, g_index, s_index, *X, Y, pool_type);
    } else if (pool_type == "MAX") {
      gather_scatter_cpu_for_loop<T, IndexT, FusedGatherScatterMaxFunctor<T>>(
          index_size, g_index, s_index, *X, Y, pool_type);
    } else if (pool_type == "MEAN") {
      gather_scatter_cpu_for_loop<T, IndexT, FusedGatherScatterSumFunctor<T>>(
          index_size, g_index, s_index, *X, Y, pool_type);
      // TODO(daisiming): Add mean operation.
    }
  }
};

}  // namespace operators
}  // namespace paddle
