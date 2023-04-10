/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/sequence_pooling.h"

#include <string>

#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/jit/kernels.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace funcs {

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = phi::EigenVector<T, MajorType, IndexType>;
template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = phi::EigenMatrix<T, MajorType, IndexType>;

template <typename T, bool is_test>
class MaxSeqPoolFunctor {
 public:
  void operator()(const phi::CPUContext& context,
                  const phi::DenseTensor& input,
                  T pad_value,
                  phi::DenseTensor* output,
                  phi::DenseTensor* index) {
    auto in_dims = input.dims();
    auto out_dims = output->dims();
    auto idx_dims = index->dims();
    PADDLE_ENFORCE_GT(in_dims.size(),
                      1,
                      errors::InvalidArgument(
                          "The rank of input shall be greater than 1, but got "
                          "the rank is %ld. Please check the input value",
                          in_dims.size()));
    PADDLE_ENFORCE_GT(out_dims.size(),
                      1,
                      errors::InvalidArgument(
                          "The rank of output shall be greater than 1, but got "
                          "the rank is %ld. Please check the input value",
                          out_dims.size()));
    for (int64_t i = 1; i < in_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          in_dims[i],
          out_dims[i],
          errors::InvalidArgument(
              "The dimension of input and output shall be same. Expected %ld "
              "== %ld, but got %ld != %ld. Please check the input value.",
              in_dims[i],
              out_dims[i],
              in_dims[i],
              out_dims[i]));
    }
    PADDLE_ENFORCE_EQ(
        idx_dims,
        out_dims,
        errors::InvalidArgument(
            "The dimension of index and output shall be same. Expected %ld == "
            "%ld, but got %ld != %ld. Please check the input value.",
            idx_dims,
            out_dims,
            idx_dims,
            out_dims));

    auto lod_level = input.lod().size();
    auto starts = input.lod()[lod_level - 1];
    const T* in_data = input.data<T>();
    T* out_data = output->data<T>();
    int* max_index = index->data<int>();

    int64_t num_seq = out_dims[0];
    int64_t dim = output->numel() / num_seq;
    for (int64_t i = 0; i < num_seq; ++i) {
      if (starts[i] == starts[i + 1]) {
        for (int64_t k = 0; k < dim; ++k) {
          out_data[i * dim + k] = pad_value;
          max_index[i * dim + k] = -1;
        }
        continue;
      }
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
  void operator()(const phi::CPUContext& context,
                  const phi::DenseTensor& input,
                  T pad_value,
                  phi::DenseTensor* output,
                  phi::DenseTensor* index) {
    auto in_dims = input.dims();
    auto out_dims = output->dims();
    PADDLE_ENFORCE_GT(in_dims.size(),
                      1,
                      errors::InvalidArgument(
                          "The rank of input shall be greater than 1, but got "
                          "%ld <= 1. Please check the input value.",
                          in_dims.size()));
    PADDLE_ENFORCE_GT(out_dims.size(),
                      1,
                      errors::InvalidArgument(
                          "The rank of output shall be greater than 1, but got "
                          "%ld <= 1. Please check the input value.",
                          out_dims.size()));
    for (int64_t i = 1; i < in_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          in_dims[i],
          out_dims[i],
          errors::InvalidArgument(
              "The dimension of input and output shall be same. Expected %ld "
              "== %ld, but got %ld != %ld. Please check the input value.",
              in_dims[i],
              out_dims[i],
              in_dims[i],
              out_dims[i]));
    }

    auto lod_level = input.lod().size();
    auto starts = input.lod()[lod_level - 1];
    const T* in_data = input.data<T>();
    T* out_data = output->data<T>();

    int64_t num_seq = out_dims[0];
    int64_t dim = output->numel() / num_seq;
    for (int64_t i = 0; i < num_seq; ++i) {
      if (starts[i] == starts[i + 1]) {
        for (int64_t k = 0; k < dim; ++k) {
          out_data[i * dim + k] = pad_value;
        }
        continue;
      }
      std::memcpy(
          &out_data[i * dim], &in_data[starts[i] * dim], dim * sizeof(T));
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
class LastSeqPoolFunctor {
 public:
  void operator()(const phi::CPUContext& context,
                  const phi::DenseTensor& input,
                  T pad_value,
                  phi::DenseTensor* output) {
    // Create pointers to input and output data
    auto* in_data = input.data<T>();
    auto* out_data = output->data<T>();

    // Calculate the size of each item in sequence
    int64_t item_size = input.numel() / input.dims()[0];
    auto lod_level = input.lod().size();
    auto lod = input.lod()[lod_level - 1];
    int seq_num = static_cast<int>(lod.size()) - 1;
    for (int i = 0; i < seq_num; ++i) {
      // Calculate the length of each sequence
      int64_t seq_len = static_cast<int64_t>(lod[i + 1] - lod[i]);
      if (seq_len == 0) {
        for (int j = 0; j < item_size; ++j) {
          out_data[j] = pad_value;
        }
      } else {
        // Point to the begin of next sequence
        in_data += seq_len * item_size;
        // Copy the last item of sequence to output
        std::memcpy(out_data, (in_data - item_size), item_size * sizeof(T));
      }
      out_data += item_size;
    }
  }
};

template <typename T>
class FirstSeqPoolFunctor {
 public:
  void operator()(const phi::CPUContext& context,
                  const phi::DenseTensor& input,
                  T pad_value,
                  phi::DenseTensor* output) {
    // Create pointers to input and output data
    auto* in_data = input.data<T>();
    auto* out_data = output->data<T>();

    // Calculate the size of each item in sequence
    int64_t item_size = input.numel() / input.dims()[0];
    auto lod_level = input.lod().size();
    auto lod = input.lod()[lod_level - 1];
    int seq_num = static_cast<int>(lod.size()) - 1;
    for (int i = 0; i < seq_num; ++i) {
      // Calculate the length of each sequence
      int64_t seq_len = static_cast<int64_t>(lod[i + 1] - lod[i]);
      if (seq_len == 0) {
        for (int j = 0; j < item_size; ++j) {
          out_data[j] = pad_value;
        }
      } else {
        // Copy the first item of sequence to output
        std::memcpy(out_data, in_data, item_size * sizeof(T));
        // Point to the next sequence
        in_data += seq_len * item_size;
      }
      out_data += item_size;
    }
  }
};

template <typename T>
class SequencePoolFunctor<phi::CPUContext, T> {
 public:
  /* max pool has index output */
  void operator()(const phi::CPUContext& context,
                  const std::string pooltype,
                  T pad_value,
                  const phi::DenseTensor& input,
                  phi::DenseTensor* output,
                  bool is_test,
                  phi::DenseTensor* index = nullptr) {
    if (pooltype == "MAX") {
      if (is_test) {
        phi::funcs::MaxSeqPoolFunctor<T, true> max_pool;
        max_pool(context, input, pad_value, output, index);
      } else {
        phi::funcs::MaxSeqPoolFunctor<T, false> max_pool;
        max_pool(context, input, pad_value, output, index);
      }
      return;
    }
    if (pooltype == "LAST") {
      phi::funcs::LastSeqPoolFunctor<T> last_pool;
      last_pool(context, input, pad_value, output);
      return;
    }
    if (pooltype == "FIRST") {
      phi::funcs::FirstSeqPoolFunctor<T> first_pool;
      first_pool(context, input, pad_value, output);
      return;
    }
    auto lod_level = input.lod().size();
    auto lod = input.lod()[lod_level - 1];
    if (pooltype == "SUM") {
      auto place = context.GetPlace();
      PADDLE_ENFORCE_EQ(
          place == phi::CPUPlace(),
          true,
          errors::InvalidArgument(
              "Sequence_pool should run on CPU Device when pooltype is SUM"));
      const T* src = input.data<T>();
      T* dst = output->mutable_data<T>(place);
      phi::jit::seq_pool_attr_t attr(
          static_cast<int>(input.numel() / input.dims()[0]),
          phi::jit::SeqPoolType::kSum);
      auto seqpool = phi::jit::KernelFuncs<phi::jit::SeqPoolTuple<T>,
                                           phi::CPUPlace>::Cache()
                         .At(attr);
      for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
        attr.h = static_cast<int>(lod[i + 1] - lod[i]);
        if (attr.h == 0) {
          for (int j = 0; j < attr.w; ++j) {
            dst[j] = pad_value;
          }
        } else {
          seqpool(src, dst, &attr);
        }
        dst += attr.w;
        src += attr.h * attr.w;
      }
      return;
    }
    auto& place = *context.eigen_device();
    for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
      phi::DenseTensor out_t = output->Slice(i, i + 1);
      int64_t w = input.numel() / input.dims()[0];
      if (lod[i] == lod[i + 1]) {
        for (int j = 0; j < w; ++j) {
          out_t.data<T>()[j] = pad_value;
        }
        continue;
      }
      phi::DenseTensor in_t =
          input.Slice(static_cast<int>(lod[i]), static_cast<int>(lod[i + 1]));
      int64_t h = static_cast<int64_t>(lod[i + 1] - lod[i]);
      auto in_e = EigenMatrix<T>::From(in_t, phi::make_ddim({h, w}));
      auto out_e = EigenVector<T>::Flatten(out_t);
      if (pooltype == "AVERAGE") {
        out_e.device(place) = in_e.mean(Eigen::array<int, 1>({{0}}));
      } else if (pooltype == "SQRT") {
        out_e.device(place) = in_e.sum(Eigen::array<int, 1>({{0}})) /
                              std::sqrt(static_cast<T>(h));
      } else {
        PADDLE_THROW(errors::InvalidArgument(
            "unsupported pooling pooltype: %s. Only support \"AVERAGE\" and "
            "\"SQRT\"",
            pooltype));
      }
    }
  }
};

template class SequencePoolFunctor<phi::CPUContext, float>;
template class SequencePoolFunctor<phi::CPUContext, double>;

}  // namespace funcs
}  // namespace phi
