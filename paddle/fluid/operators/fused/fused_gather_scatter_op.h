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
  void operator()(const bool& first_flag, const Tensor& src_slice,
                  Tensor* dst_slice) {
    auto eigen_src = framework::EigenVector<T>::Flatten(src_slice);
    auto eigen_dst = framework::EigenVector<T>::Flatten(*dst_slice);
    eigen_dst += eigen_src;
  }
};

template <typename T>
struct FusedGatherScatterMinFunctor {
  void operator()(const bool& first_flag, const Tensor& src_slice,
                  Tensor* dst_slice) {
    auto eigen_src = framework::EigenVector<T>::Flatten(src_slice);
    auto eigen_dst = framework::EigenVector<T>::Flatten(*dst_slice);
    if (first_flag) {
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
    if (first_flag) {
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
void gather_scatter_cpu_for_loop(const int& input_size, const int& index_size,
                                 const IndexT* g_index, const IndexT* s_index,
                                 const Tensor& src, Tensor* dst,
                                 const std::string& pool_type,
                                 int* scatter_count = NULL) {
  Functor functor;
  if (pool_type == "SUM") {
    for (int i = 0; i < index_size; ++i) {
      IndexT src_idx = g_index[i];
      IndexT dst_idx = s_index[i];
      elementwise_inner_operation<T, IndexT, Functor>(src, dst, src_idx,
                                                      dst_idx, false, functor);
    }
  } else if (pool_type == "MEAN") {
    for (int i = 0; i < index_size; ++i) {
      IndexT src_idx = g_index[i];
      IndexT dst_idx = s_index[i];
      elementwise_inner_operation<T, IndexT, Functor>(src, dst, src_idx,
                                                      dst_idx, false, functor);
    }
    for (int i = 0; i < index_size; ++i) {
      IndexT dst_idx = s_index[i];
      *(scatter_count + dst_idx) += 1;
    }
    for (int i = 0; i < input_size; ++i) {
      if (*(scatter_count + i) == 0) continue;
      auto dst_slice = dst->Slice(i, i + 1);
      auto eigen_dst = framework::EigenVector<T>::Flatten(dst_slice);
      eigen_dst = eigen_dst / static_cast<T>(*(scatter_count + i));
    }
  } else if (pool_type == "MIN" || pool_type == "MAX") {
    std::set<IndexT> existed_dst;
    for (int i = 0; i < index_size; ++i) {
      IndexT src_idx = g_index[i];
      IndexT dst_idx = s_index[i];
      bool in_set = existed_dst.find(dst_idx) != existed_dst.end();
      if (!in_set) {
        elementwise_inner_operation<T, IndexT, Functor>(src, dst, src_idx,
                                                        dst_idx, true, functor);
        existed_dst.emplace(dst_idx);
      } else {
        elementwise_inner_operation<T, IndexT, Functor>(
            src, dst, src_idx, dst_idx, false, functor);
      }
    }
  }
}

template <typename T, typename IndexT, typename Functor>
void gather_scatter_cpu_for_loop_grad(
    const int& input_size, const int& index_size, const IndexT* g_index,
    const IndexT* s_index, const Tensor& src, Tensor* dst,
    const std::string& pool_type, const int* scatter_count = nullptr,
    const Tensor* input = nullptr, const Tensor* output = nullptr) {
  if (pool_type == "SUM") {
    Functor functor;
    for (int i = 0; i < index_size; ++i) {
      IndexT src_idx = g_index[i];
      IndexT dst_idx = s_index[i];
      elementwise_inner_operation<T, IndexT, Functor>(src, dst, src_idx,
                                                      dst_idx, false, functor);
    }
  } else if (pool_type == "MEAN") {
    for (int i = 0; i < index_size; ++i) {
      IndexT src_idx = g_index[i];
      IndexT dst_idx = s_index[i];
      auto src_slice = src.Slice(src_idx, src_idx + 1);
      auto dst_slice = dst->Slice(dst_idx, dst_idx + 1);
      auto eigen_src = framework::EigenVector<T>::Flatten(src_slice);
      auto eigen_dst = framework::EigenVector<T>::Flatten(dst_slice);
      eigen_dst += (eigen_src / static_cast<T>(scatter_count[src_idx]));
    }
  } else if (pool_type == "MIN" || pool_type == "MAX") {
    for (int i = 0; i < index_size; ++i) {
      auto forward_src_idx = s_index[i];
      auto forward_dst_idx = g_index[i];
      auto input_slice = input->Slice(forward_src_idx, forward_src_idx + 1);
      auto output_slice = output->Slice(forward_dst_idx, forward_dst_idx + 1);
      auto eigen_input = framework::EigenVector<T>::Flatten(input_slice);
      auto eigen_output = framework::EigenVector<T>::Flatten(output_slice);

      auto src_slice = src.Slice(forward_dst_idx, forward_dst_idx + 1);
      auto dst_slice = dst->Slice(forward_src_idx, forward_src_idx + 1);
      auto eigen_src = framework::EigenVector<T>::Flatten(src_slice);
      auto eigen_dst = framework::EigenVector<T>::Flatten(dst_slice);
      eigen_dst += eigen_src * (eigen_output == eigen_input);
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

    const int& index_size = gather_index->dims()[0];
    if (index_size == 0) return;

    T* p_output = Y->mutable_data<T>(ctx.GetPlace());
    const auto& src_dims = X->dims();
    int64_t memset_size = 1;
    for (int i = 0; i < src_dims.size(); ++i) memset_size *= src_dims[i];
    const size_t& memset_bytes = memset_size * sizeof(T);
    memset(p_output, 0, memset_bytes);

    const IndexT* g_index = gather_index->data<IndexT>();
    const IndexT* s_index = scatter_index->data<IndexT>();

    const std::string& pool_type = ctx.Attr<std::string>("pool_type");
    if (pool_type == "SUM") {
      gather_scatter_cpu_for_loop<T, IndexT, FusedGatherScatterSumFunctor<T>>(
          src_dims[0], index_size, g_index, s_index, *X, Y, pool_type);
    } else if (pool_type == "MIN") {
      gather_scatter_cpu_for_loop<T, IndexT, FusedGatherScatterMinFunctor<T>>(
          src_dims[0], index_size, g_index, s_index, *X, Y, pool_type);
    } else if (pool_type == "MAX") {
      gather_scatter_cpu_for_loop<T, IndexT, FusedGatherScatterMaxFunctor<T>>(
          src_dims[0], index_size, g_index, s_index, *X, Y, pool_type);
    } else if (pool_type == "MEAN") {
      auto* scatter_count = ctx.Output<Tensor>("Scatter_count");
      int* p_scatter_count = scatter_count->mutable_data<int>(ctx.GetPlace());
      memset(p_scatter_count, 0, src_dims[0] * sizeof(int));
      gather_scatter_cpu_for_loop<T, IndexT, FusedGatherScatterSumFunctor<T>>(
          src_dims[0], index_size, g_index, s_index, *X, Y, pool_type,
          p_scatter_count);
    }
  }
};

template <typename DeviceContext, typename T, typename IndexT>
class FusedGatherScatterGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* gather_index = ctx.Input<Tensor>("Scatter_index");
    auto* scatter_index = ctx.Input<Tensor>("Gather_index");
    auto* Y = ctx.Output<Tensor>(framework::GradVarName("X"));

    const int& index_size = gather_index->dims()[0];
    if (index_size == 0) return;

    T* p_output = Y->mutable_data<T>(ctx.GetPlace());
    const auto& src_dims = X->dims();
    int64_t memset_size = 1;
    for (int i = 0; i < src_dims.size(); ++i) memset_size *= src_dims[i];
    const size_t& memset_bytes = memset_size * sizeof(T);
    memset(p_output, 0, memset_bytes);

    const IndexT* g_index = gather_index->data<IndexT>();
    const IndexT* s_index = scatter_index->data<IndexT>();

    const std::string& pool_type = ctx.Attr<std::string>("pool_type");
    if (pool_type == "SUM") {
      gather_scatter_cpu_for_loop_grad<T, IndexT,
                                       FusedGatherScatterSumFunctor<T>>(
          src_dims[0], index_size, g_index, s_index, *X, Y, pool_type);
    } else if (pool_type == "MEAN") {
      auto* scatter_count = ctx.Input<Tensor>("Scatter_count");
      const int* s_count = scatter_count->data<int>();
      gather_scatter_cpu_for_loop_grad<T, IndexT,
                                       FusedGatherScatterSumFunctor<T>>(
          src_dims[0], index_size, g_index, s_index, *X, Y, pool_type, s_count);
    } else if (pool_type == "MIN" || pool_type == "MAX") {
      auto* input = ctx.Input<Tensor>("X");
      auto* output = ctx.Input<Tensor>("Out");
      // Functor not used here.
      gather_scatter_cpu_for_loop_grad<T, IndexT,
                                       FusedGatherScatterMinFunctor<T>>(
          src_dims[0], index_size, g_index, s_index, *X, Y, pool_type, nullptr,
          input, output);
    }
  }
};

}  // namespace operators
}  // namespace paddle
