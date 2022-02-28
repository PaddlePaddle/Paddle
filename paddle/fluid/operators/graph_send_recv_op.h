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
struct GraphSendRecvSumFunctor {
  void operator()(const bool& first_flag, const Tensor& src_slice,
                  Tensor* dst_slice) {
    auto eigen_src = framework::EigenVector<T>::Flatten(src_slice);
    auto eigen_dst = framework::EigenVector<T>::Flatten(*dst_slice);
    eigen_dst += eigen_src;
  }
};

template <typename T>
struct GraphSendRecvMinFunctor {
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
struct GraphSendRecvMaxFunctor {
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
void graph_send_recv_cpu_for_loop(const int& input_size, const int& index_size,
                                  const IndexT* s_index, const IndexT* d_index,
                                  const Tensor& src, Tensor* dst,
                                  const std::string& pool_type,
                                  int* dst_count = nullptr) {
  Functor functor;
  if (pool_type == "SUM") {
    for (int i = 0; i < index_size; ++i) {
      const IndexT& src_idx = s_index[i];
      const IndexT& dst_idx = d_index[i];
      elementwise_inner_operation<T, IndexT, Functor>(src, dst, src_idx,
                                                      dst_idx, false, functor);
    }
  } else if (pool_type == "MEAN") {
    for (int i = 0; i < index_size; ++i) {
      const IndexT& src_idx = s_index[i];
      const IndexT& dst_idx = d_index[i];
      elementwise_inner_operation<T, IndexT, Functor>(src, dst, src_idx,
                                                      dst_idx, false, functor);
    }
    for (int i = 0; i < index_size; ++i) {
      IndexT dst_idx = d_index[i];
      *(dst_count + dst_idx) += 1;
    }
    for (int i = 0; i < input_size; ++i) {
      if (*(dst_count + i) == 0) continue;
      auto dst_slice = dst->Slice(i, i + 1);
      auto eigen_dst = framework::EigenVector<T>::Flatten(dst_slice);
      eigen_dst = eigen_dst / static_cast<T>(*(dst_count + i));
    }
  } else if (pool_type == "MIN" || pool_type == "MAX") {
    std::set<IndexT> existed_dst;
    for (int i = 0; i < index_size; ++i) {
      const IndexT& src_idx = s_index[i];
      const IndexT& dst_idx = d_index[i];
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
void graph_send_recv_cpu_for_loop_grad(
    const int& input_size, const int& index_size, const IndexT* s_index,
    const IndexT* d_index, const Tensor& src, Tensor* dst,
    const std::string& pool_type, const int* dst_count = nullptr,
    const Tensor* input = nullptr, const Tensor* output = nullptr) {
  if (pool_type == "SUM") {
    Functor functor;
    for (int i = 0; i < index_size; ++i) {
      const IndexT& src_idx = s_index[i];
      const IndexT& dst_idx = d_index[i];
      elementwise_inner_operation<T, IndexT, Functor>(src, dst, src_idx,
                                                      dst_idx, false, functor);
    }
  } else if (pool_type == "MEAN") {
    for (int i = 0; i < index_size; ++i) {
      const IndexT& src_idx = s_index[i];
      const IndexT& dst_idx = d_index[i];
      auto src_slice = src.Slice(src_idx, src_idx + 1);
      auto dst_slice = dst->Slice(dst_idx, dst_idx + 1);
      auto eigen_src = framework::EigenVector<T>::Flatten(src_slice);
      auto eigen_dst = framework::EigenVector<T>::Flatten(dst_slice);
      eigen_dst += (eigen_src / static_cast<T>(dst_count[src_idx]));
    }
  } else if (pool_type == "MIN" || pool_type == "MAX") {
    for (int i = 0; i < index_size; ++i) {
      const IndexT& forward_src_idx = d_index[i];
      const IndexT& forward_dst_idx = s_index[i];
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
void GraphSendRecvOpKernelLaunchHelper(const framework::ExecutionContext& ctx,
                                       const Tensor& src_index) {
  auto* X = ctx.Input<Tensor>("X");
  auto* dst_index = ctx.Input<Tensor>("Dst_index");
  auto* Y = ctx.Output<Tensor>("Out");

  const int& index_size = src_index.dims()[0];

  T* p_output = Y->mutable_data<T>(ctx.GetPlace());
  const auto& src_dims = X->dims();
  int64_t memset_size = 1;
  for (int i = 0; i < src_dims.size(); ++i) memset_size *= src_dims[i];
  const size_t& memset_bytes = memset_size * sizeof(T);
  memset(p_output, 0, memset_bytes);

  if (index_size == 0) return;

  const IndexT* s_index = src_index.data<IndexT>();
  const IndexT* d_index = dst_index->data<IndexT>();
  const std::string& pool_type = ctx.Attr<std::string>("pool_type");
  if (pool_type == "SUM") {
    graph_send_recv_cpu_for_loop<T, IndexT, GraphSendRecvSumFunctor<T>>(
        src_dims[0], index_size, s_index, d_index, *X, Y, pool_type);
  } else if (pool_type == "MIN") {
    graph_send_recv_cpu_for_loop<T, IndexT, GraphSendRecvMinFunctor<T>>(
        src_dims[0], index_size, s_index, d_index, *X, Y, pool_type);
  } else if (pool_type == "MAX") {
    graph_send_recv_cpu_for_loop<T, IndexT, GraphSendRecvMaxFunctor<T>>(
        src_dims[0], index_size, s_index, d_index, *X, Y, pool_type);
  } else if (pool_type == "MEAN") {
    auto* dst_count = ctx.Output<Tensor>("Dst_count");
    int* p_dst_count = dst_count->mutable_data<int>(ctx.GetPlace());
    memset(p_dst_count, 0, src_dims[0] * sizeof(int));
    graph_send_recv_cpu_for_loop<T, IndexT, GraphSendRecvSumFunctor<T>>(
        src_dims[0], index_size, s_index, d_index, *X, Y, pool_type,
        p_dst_count);
  }
}

template <typename DeviceContext, typename T, typename IndexT>
void GraphSendRecvGradOpKernelLaunchHelper(
    const framework::ExecutionContext& ctx, const Tensor& src_index) {
  auto* X = ctx.Input<Tensor>(framework::GradVarName("Out"));
  auto* dst_index = ctx.Input<Tensor>("Src_index");
  auto* Y = ctx.Output<Tensor>(framework::GradVarName("X"));

  const int& index_size = src_index.dims()[0];

  T* p_output = Y->mutable_data<T>(ctx.GetPlace());
  const auto& src_dims = X->dims();
  int64_t memset_size = 1;
  for (int i = 0; i < src_dims.size(); ++i) memset_size *= src_dims[i];
  const size_t& memset_bytes = memset_size * sizeof(T);
  memset(p_output, 0, memset_bytes);

  if (index_size == 0) return;

  const IndexT* s_index = src_index.data<IndexT>();
  const IndexT* d_index = dst_index->data<IndexT>();

  const std::string& pool_type = ctx.Attr<std::string>("pool_type");
  if (pool_type == "SUM") {
    graph_send_recv_cpu_for_loop_grad<T, IndexT, GraphSendRecvSumFunctor<T>>(
        src_dims[0], index_size, s_index, d_index, *X, Y, pool_type);
  } else if (pool_type == "MEAN") {
    auto* dst_count = ctx.Input<Tensor>("Dst_count");
    const int* s_count = dst_count->data<int>();
    // Functor not used here.
    graph_send_recv_cpu_for_loop_grad<T, IndexT, GraphSendRecvSumFunctor<T>>(
        src_dims[0], index_size, s_index, d_index, *X, Y, pool_type, s_count);
  } else if (pool_type == "MIN" || pool_type == "MAX") {
    const auto* input = ctx.Input<Tensor>("X");
    const auto* output = ctx.Input<Tensor>("Out");
    // Functor not used here.
    graph_send_recv_cpu_for_loop_grad<T, IndexT, GraphSendRecvMinFunctor<T>>(
        src_dims[0], index_size, s_index, d_index, *X, Y, pool_type, nullptr,
        input, output);
  }
}

template <typename DeviceContext, typename T>
class GraphSendRecvOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* src_index = ctx.Input<Tensor>("Src_index");
    auto index_type = framework::TransToProtoVarType(src_index->dtype());

    if (index_type == framework::proto::VarType::INT32) {
      GraphSendRecvOpKernelLaunchHelper<DeviceContext, T, int>(ctx, *src_index);
    } else if (index_type == framework::proto::VarType::INT64) {
      GraphSendRecvOpKernelLaunchHelper<DeviceContext, T, int64_t>(ctx,
                                                                   *src_index);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupported Src_index or Dst_index type, Expected int, int64, but "
          "got %s.",
          index_type));
    }
  }
};

template <typename DeviceContext, typename T>
class GraphSendRecvGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* src_index = ctx.Input<Tensor>("Dst_index");
    auto index_type = framework::TransToProtoVarType(src_index->dtype());

    if (index_type == framework::proto::VarType::INT32) {
      GraphSendRecvGradOpKernelLaunchHelper<DeviceContext, T, int>(ctx,
                                                                   *src_index);
    } else if (index_type == framework::proto::VarType::INT64) {
      GraphSendRecvGradOpKernelLaunchHelper<DeviceContext, T, int64_t>(
          ctx, *src_index);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupported Src_index or Dst_index type, Expected int, int64, but "
          "got %s.",
          index_type));
    }
  }
};

}  // namespace operators
}  // namespace paddle
