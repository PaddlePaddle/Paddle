// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.1
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#include "paddle/fluid/operators/kernel_primitives/kernel_primitives.h"

namespace paddle {
namespace operators {

namespace kps = paddle::operators::kernel_primitives;

struct DimensionsTransform {
  using DimVector = std::vector<int64_t>;
  typedef void (*MergeFunctor)(bool &, std::vector<DimVector> &, DimVector &,
                               int, int);
  int64_t dim_size;
  DimVector out_dims;
  std::vector<DimVector> in_dims;

 private:
  // To compensate the lackage of input_tensors` dimension with input variable
  // 'axis'
  void InputDimensionsExtend(int N, int axis) {
    for (auto &in_dim : in_dims) {
      int64_t in_idx = 0;
      if (in_dim.size() < dim_size) {
        DimVector tmp_dim(dim_size, 1);
        do {
          if (in_dim[in_idx] == out_dims[axis] || in_dim[in_idx] == 1) {
            tmp_dim[axis] = in_dim[in_idx];
            in_idx++;
            axis++;
          } else {
            PADDLE_THROW(platform::errors::InvalidArgument(
                "The %d-th dimension of input tensor is expected to be equal "
                "with the %d-th dimension of output tensor %d or 1, but "
                "recieved %d.",
                in_idx + 1, axis + 1, out_dims[axis], in_dim[in_idx]));
          }
        } while (in_idx < in_dim.size());
        in_dim.resize(dim_size);
        std::copy(tmp_dim.begin(), tmp_dim.end(), in_dim.begin());
      } else {
        do {
          if (in_dim[in_idx] == out_dims[in_idx] || in_dim[in_idx] == 1) {
            in_idx++;
          } else {
            PADDLE_THROW(platform::errors::InvalidArgument(
                "The %d-th dimension of input tensor is expected to be equal "
                "with the %d-th dimension of output tensor %d or 1, but "
                "recieved %d.",
                in_idx + 1, in_idx + 1, out_dims[in_idx], in_dim[in_idx]));
          }
        } while (in_idx < dim_size);
      }
      std::reverse(in_dim.begin(), in_dim.end());
    }
    std::reverse(out_dims.begin(), out_dims.end());
  }

  template <typename MergeFunctor>
  __inline__ void MergeDimensions(MergeFunctor merge_func, int N) {
    auto VectorReorganise = [](DimVector *vec, int l_idx, int m_idx) {
      (*vec)[m_idx - 1] =
          std::accumulate(vec->begin() + l_idx, vec->begin() + m_idx, 1,
                          std::multiplies<int64_t>());
      vec->erase(vec->begin() + l_idx, vec->begin() + m_idx - 1);
    };

    int64_t i = 0;
    while (i < dim_size) {
      int cnt = 0;
      int low_idx = i;
      bool equal = true;
      do {
        merge_func(equal, in_dims, out_dims, i, N);
        if (equal) {
          i++;
          cnt++;
        } else {
          break;
        }
      } while (i < dim_size);

      if (cnt > 1) {
        for (auto &in_dim : in_dims) {
          VectorReorganise(&in_dim, low_idx, i);
        }
        VectorReorganise(&out_dims, low_idx, i);
        dim_size -= --cnt;
        i -= cnt;
      } else if (cnt < 1) {
        i++;
      }
    }
  }

 public:
  explicit DimensionsTransform(
      const std::vector<const framework::Tensor *> &ins,
      const framework::DDim &dims, int axis) {
    const int N = ins.size();
    dim_size = dims.size();
    out_dims = framework::vectorize<int64_t>(dims);
    in_dims.resize(N);
    for (int j = 0; j < N; ++j) {
      in_dims[j] = framework::vectorize<int64_t>(ins[j]->dims());
    }
    InputDimensionsExtend(N, axis);

    auto merge_sequential_dims = [](bool &equal,
                                    std::vector<DimVector> &in_dims,
                                    DimVector &out, int i, int num) {
      for (int j = 1; j < num; ++j) {
        equal = (in_dims[0][i] == in_dims[j][i]) ? true : false;
      }
    };
    auto merge_sequential_one_dims = [](bool &equal,
                                        std::vector<DimVector> &in_dims,
                                        DimVector &out, int i, int num) {
      equal = in_dims[0][i] == 1;
      if (equal) {
        for (int j = 1; j < num; ++j) {
          equal = in_dims[j][i] == out[i];
        }
      }
    };
    // To Merge the dimensions of input_tensors while the consequtive
    // equal-dimensions appears.
    MergeFunctor merge_ptr = merge_sequential_dims;
    MergeDimensions<MergeFunctor>(merge_ptr, N);

    int min_idx = 0;
    int min_val = std::accumulate(in_dims[0].begin(), in_dims[0].end(), 1,
                                  std::multiplies<int64_t>());
    for (int j = 1; j < N; ++j) {
      int temp = std::accumulate(in_dims[j].begin(), in_dims[j].end(), 1,
                                 std::multiplies<int64_t>());
      min_val = min_val > temp ? temp : min_val;
      min_idx = min_val == temp ? j : min_idx;
    }
    std::swap(in_dims[0], in_dims[min_idx]);

    // To Merge the dimension of input_tensors while the consequtive
    // 1-value-dimensions appears.
    merge_ptr = merge_sequential_one_dims;
    MergeDimensions<MergeFunctor>(merge_ptr, N);
    std::swap(in_dims[min_idx], in_dims[0]);
  }
};

template <ElementwiseType ET, typename InT, typename OutT, typename Functor>
void LaunchBroadcastElementwiseCudaKernel(
    const platform::CUDADeviceContext &ctx,
    const std::vector<const framework::Tensor *> &ins,
    std::vector<framework::Tensor *> *outs, int axis, Functor func) {
  std::vector<const pten::DenseTensor *> pt_inputs;
  std::vector<pten::DenseTensor *> pt_outputs;
  // TODO(YuanRisheng) *_tmp for cache DenseTensor, because the temporary
  // DenseTensor obj
  // generated by MakePtenDenseTensor can be destroyed when exits loop. *_tmp
  // can be deleted
  // when DenseTensor support copy constructor.
  std::vector<std::unique_ptr<pten::DenseTensor>> pt_inputs_tmp;
  std::vector<std::unique_ptr<pten::DenseTensor>> pt_outputs_tmp;
  for (auto in : ins) {
    pt_inputs_tmp.emplace_back(
        std::move(paddle::experimental::MakePtenDenseTensor(*in)));
  }
  for (auto out : *outs) {
    pt_outputs_tmp.emplace_back(
        std::move(paddle::experimental::MakePtenDenseTensor(*out)));
  }
  for (int i = 0; i < pt_inputs_tmp.size(); i++) {
    pt_inputs.push_back(pt_inputs_tmp[i].get());
  }
  for (int i = 0; i < pt_outputs_tmp.size(); i++) {
    pt_outputs.push_back(pt_outputs_tmp[i].get());
  }
  pten::LaunchBroadcastElementwiseCudaKernel<ET, InT, OutT>(
      ctx, pt_inputs, &pt_outputs, axis, func);
}

template <ElementwiseType ET, typename InT, typename OutT, typename Functor>
void LaunchElementwiseCudaKernel(
    const platform::CUDADeviceContext &cuda_ctx,
    const std::vector<const framework::Tensor *> &ins,
    std::vector<framework::Tensor *> *outs, int axis, Functor func) {
  std::vector<const pten::DenseTensor *> pt_inputs;
  std::vector<pten::DenseTensor *> pt_outputs;
  // TODO(YuanRisheng) *_tmp for cache DenseTensor, because the temporary
  // DenseTensor obj
  // generated by MakePtenDenseTensor can be destroyed when exits loop. *_tmp
  // can be deleted
  // when DenseTensor support copy constructor.
  std::vector<std::unique_ptr<pten::DenseTensor>> pt_inputs_tmp;
  std::vector<std::unique_ptr<pten::DenseTensor>> pt_outputs_tmp;
  for (auto in : ins) {
    pt_inputs_tmp.emplace_back(
        std::move(paddle::experimental::MakePtenDenseTensor(*in)));
  }
  for (auto out : *outs) {
    pt_outputs_tmp.emplace_back(
        std::move(paddle::experimental::MakePtenDenseTensor(*out)));
  }
  for (int i = 0; i < pt_inputs_tmp.size(); i++) {
    pt_inputs.push_back(pt_inputs_tmp[i].get());
  }
  for (int i = 0; i < pt_outputs_tmp.size(); i++) {
    pt_outputs.push_back(pt_outputs_tmp[i].get());
  }
  pten::LaunchElementwiseCudaKernel<ET, InT, OutT>(cuda_ctx, pt_inputs,
                                                   &pt_outputs, axis, func);
}

}  // namespace operators
}  // namespace paddle
