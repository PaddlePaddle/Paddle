/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/sequence_pooling.h"

namespace paddle {
namespace operators {
namespace math {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T, bool is_test>
class MaxSeqPoolFunctor {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::LoDTensor& input, framework::Tensor* output,
                  framework::Tensor* index) {
    auto in_dims = input.dims();
    auto out_dims = output->dims();
    auto idx_dims = index->dims();
    PADDLE_ENFORCE_GT(in_dims.size(), 1);
    PADDLE_ENFORCE_GT(out_dims.size(), 1);
    for (int64_t i = 1; i < in_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(in_dims[i], out_dims[i]);
    }
    PADDLE_ENFORCE_EQ(idx_dims, out_dims);

    auto starts = input.lod()[0];
    const T* in_data = input.data<T>();
    T* out_data = output->data<T>();
    int* max_index = index->data<int>();

    int64_t num_seq = out_dims[0];
    int64_t dim = output->numel() / num_seq;
    for (int64_t i = 0; i < num_seq; ++i) {
      for (int64_t k = 0; k < dim; ++k) {
        out_data[i * dim + k] = in_data[starts[i] * dim + k];
        max_index[i * dim + k] = starts[i];
      }
      for (size_t j = starts[i] + 1; j < starts[i + 1]; ++j) {
        for (int64_t k = 0; k < dim; ++k) {
          if (in_data[j * dim + k] > out_data[i * dim + k]) {
            out_data[i * dim + k] = in_data[j * dim + k];
            max_index[i * dim + k] = j;
          }
        }
      }
    }
  }
};
// Instantisation of Max Sequence Pooling for test phase eg. no need to fill
// index buffer
template <typename T>
class MaxSeqPoolFunctor<T, true> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::LoDTensor& input, framework::Tensor* output,
                  framework::Tensor* index) {
    auto in_dims = input.dims();
    auto out_dims = output->dims();
    PADDLE_ENFORCE_GT(in_dims.size(), 1);
    PADDLE_ENFORCE_GT(out_dims.size(), 1);
    for (int64_t i = 1; i < in_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(in_dims[i], out_dims[i]);
    }

    auto starts = input.lod()[0];
    const T* in_data = input.data<T>();
    T* out_data = output->data<T>();

    int64_t num_seq = out_dims[0];
    int64_t dim = output->numel() / num_seq;
    for (int64_t i = 0; i < num_seq; ++i) {
      std::memcpy(&out_data[i * dim], &in_data[starts[i] * dim],
                  dim * sizeof(T));
      for (size_t j = starts[i] + 1; j < starts[i + 1]; ++j) {
        for (int64_t k = 0; k < dim; ++k) {
          if (in_data[j * dim + k] > out_data[i * dim + k]) {
            out_data[i * dim + k] = in_data[j * dim + k];
          }
        }
      }
    }
  }
};
template <typename T>
class MaxSeqPoolGradFunctor {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& out_grad,
                  const framework::Tensor& index,
                  framework::LoDTensor* in_grad) {
    auto og_dims = out_grad.dims();
    auto ig_dims = in_grad->dims();
    auto idx_dims = index.dims();
    PADDLE_ENFORCE_GT(og_dims.size(), 1);
    PADDLE_ENFORCE_GT(ig_dims.size(), 1);
    for (int64_t i = 1; i < og_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(og_dims[i], ig_dims[i]);
    }
    PADDLE_ENFORCE_EQ(idx_dims, og_dims);

    const T* og_data = out_grad.data<T>();
    const int* max_index = index.data<int>();
    T* ig_data = in_grad->data<T>();

    SetConstant<platform::CPUDeviceContext, T> set_zero;
    set_zero(context, in_grad, static_cast<T>(0.0));
    int64_t num_seq = og_dims[0];
    int64_t dim = out_grad.numel() / num_seq;
    for (int64_t i = 0; i < num_seq; ++i) {
      for (int64_t j = 0; j < dim; ++j) {
        int step_id = max_index[i * dim + j];
        ig_data[step_id * dim + j] = og_data[i * dim + j];
      }
    }
  }
};

template <typename T>
class LastSeqPoolFunctor {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::LoDTensor& input,
                  framework::Tensor* output) {
    // Create pointers to input and output data
    auto* in_data = input.data<T>();
    auto* out_data = output->data<T>();

    // Calculate the size of each item in sequence
    int64_t item_size = input.numel() / input.dims()[0];
    auto lod = input.lod()[0];
    int seq_num = static_cast<int>(lod.size()) - 1;
    for (int i = 0; i < seq_num; ++i) {
      // Calculate the length of each sequence
      int64_t seq_len = static_cast<int64_t>(lod[i + 1] - lod[i]);
      // Point to the begin of next sequence
      in_data += seq_len * item_size;
      // Copy the last item of sequence to output
      std::memcpy(out_data, (in_data - item_size), item_size * sizeof(T));
      out_data += item_size;
    }
  }
};

template <typename T>
class FirstSeqPoolFunctor {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::LoDTensor& input,
                  framework::Tensor* output) {
    // Create pointers to input and output data
    auto* in_data = input.data<T>();
    auto* out_data = output->data<T>();

    // Calculate the size of each item in sequence
    int64_t item_size = input.numel() / input.dims()[0];
    auto lod = input.lod()[0];
    int seq_num = static_cast<int>(lod.size()) - 1;
    for (int i = 0; i < seq_num; ++i) {
      // Calculate the length of each sequence
      int64_t seq_len = static_cast<int64_t>(lod[i + 1] - lod[i]);
      // Copy the first item of sequence to output
      std::memcpy(out_data, in_data, item_size * sizeof(T));
      // Point to the next sequence
      in_data += seq_len * item_size;
      out_data += item_size;
    }
  }
};

template <typename T>
class SumSeqPoolGradFunctor {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& out_grad,
                  framework::LoDTensor* in_grad) {
    auto lod = in_grad->lod()[0];
    int64_t out_w = out_grad.numel() / out_grad.dims()[0];
    int64_t in_w = in_grad->numel() / in_grad->dims()[0];
    PADDLE_ENFORCE(in_w == out_w);
    const T* out_g_data = out_grad.data<T>();
    T* in_g_data = in_grad->mutable_data<T>(context.GetPlace());
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
    for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
      int64_t h = static_cast<int64_t>(lod[i + 1] - lod[i]);
      int64_t in_offset = lod[i] * in_w;
      const T* out_pos = out_g_data + i * out_w;
      T* in_pos = in_g_data + in_offset;
      for (int r = 0; r != h; ++r) {
        blas.VCOPY(in_w, out_pos, in_pos + r * in_w);
      }
    }
  }
};

template <typename T>
class SequencePoolFunctor<platform::CPUDeviceContext, T> {
 public:
  /* max pool has index output */
  void operator()(const platform::CPUDeviceContext& context,
                  const std::string pooltype, const framework::LoDTensor& input,
                  framework::Tensor* output, bool is_test,
                  framework::Tensor* index = nullptr) {
    if (pooltype == "MAX") {
      if (is_test) {
        math::MaxSeqPoolFunctor<T, true> max_pool;
        max_pool(context, input, output, index);
      } else {
        math::MaxSeqPoolFunctor<T, false> max_pool;
        max_pool(context, input, output, index);
      }
      return;
    }
    if (pooltype == "LAST") {
      math::LastSeqPoolFunctor<T> last_pool;
      last_pool(context, input, output);
      return;
    }

    if (pooltype == "FIRST") {
      math::FirstSeqPoolFunctor<T> first_pool;
      first_pool(context, input, output);
      return;
    }
    auto lod = input.lod()[0];
    auto& place = *context.eigen_device();
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
    for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
      Tensor in_t =
          input.Slice(static_cast<int>(lod[i]), static_cast<int>(lod[i + 1]));
      Tensor out_t = output->Slice(i, i + 1);
      int64_t h = static_cast<int64_t>(lod[i + 1] - lod[i]);
      int64_t w = input.numel() / input.dims()[0];
      auto in_e = EigenMatrix<T>::From(in_t, framework::make_ddim({h, w}));
      auto out_e = EigenVector<T>::Flatten(out_t);
      if (pooltype == "AVERAGE") {
        out_e.device(place) = in_e.mean(Eigen::array<int, 1>({{0}}));
      } else if (pooltype == "SUM") {
        if (h > 0) {
          const T* in_data = in_t.data<T>();
          T* out_data = out_t.mutable_data<T>(context.GetPlace());
          blas.VCOPY(w, in_data, out_data);
          for (int64_t r = 1; r != h; ++r) {
            blas.AXPY(w, 1., in_data + r * w, out_data);
          }
        }
      } else if (pooltype == "SQRT") {
        out_e.device(place) = in_e.sum(Eigen::array<int, 1>({{0}})) /
                              std::sqrt(static_cast<T>(h));
      } else {
        PADDLE_THROW("unsupported pooling pooltype");
      }
    }
  }
};

template <typename T>
class SequencePoolGradFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const std::string pooltype, const framework::Tensor& out_grad,
                  framework::LoDTensor* in_grad,
                  /* max pool has index */
                  const framework::Tensor* index = nullptr) {
    if (pooltype == "MAX") {
      math::MaxSeqPoolGradFunctor<T> max_pool_grad;
      max_pool_grad(context, out_grad, *index, in_grad);
      return;
    }

    if (pooltype == "LAST" || pooltype == "FIRST") {
      // set X@Grad be zero at first when pooltype is LAST/FIRST
      math::SetConstant<platform::CPUDeviceContext, T> functor;
      functor(context, in_grad, 0);
    }

    if (pooltype == "SUM") {
      math::SumSeqPoolGradFunctor<T> sum_pool_grad;
      sum_pool_grad(context, out_grad, in_grad);
      return;
    }

    auto lod = in_grad->lod()[0];
    auto& place = *context.eigen_device();
    for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
      auto in_g_t = in_grad->Slice(static_cast<int>(lod[i]),
                                   static_cast<int>(lod[i + 1]));
      auto out_g_t = out_grad.Slice(i, i + 1);
      int64_t h = static_cast<int64_t>(lod[i + 1] - lod[i]);
      int64_t w = in_grad->numel() / in_grad->dims()[0];
      auto in_g_e = EigenMatrix<T>::From(in_g_t, {h, w});
      auto out_g_e = EigenMatrix<T>::From(out_g_t, {1, w});
      auto out_g_e_v = EigenVector<T>::Flatten(out_g_t);
      Eigen::DSizes<int, 2> bcast(h, 1);

      if (pooltype == "AVERAGE") {
        in_g_e.device(place) = (out_g_e / static_cast<T>(h)).broadcast(bcast);
      } else if (pooltype == "SQRT") {
        in_g_e.device(place) =
            (out_g_e / std::sqrt(static_cast<T>(h))).broadcast(bcast);
      } else if (pooltype == "LAST") {
        in_g_e.chip(h - 1, 0).device(place) = out_g_e_v;
      } else if (pooltype == "FIRST") {
        in_g_e.chip(0, 0).device(place) = out_g_e_v;
      } else {
        PADDLE_THROW("unsupported pooling pooltype");
      }
    }
  }
};

template class SequencePoolFunctor<platform::CPUDeviceContext, float>;
template class SequencePoolFunctor<platform::CPUDeviceContext, double>;
template class SequencePoolGradFunctor<platform::CPUDeviceContext, float>;
template class SequencePoolGradFunctor<platform::CPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
