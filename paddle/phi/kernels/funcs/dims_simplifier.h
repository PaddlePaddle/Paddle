/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace funcs {

struct BroadcastDimsSimplifier {
  using DimVector = std::vector<int64_t>;
  typedef void (*MergeFunctor)(
      bool &, std::vector<DimVector> &, DimVector &, int, int);

  int64_t N;
  int64_t rank;
  DimVector out_dims;
  std::vector<DimVector> in_dims;

 public:
  BroadcastDimsSimplifier(const std::vector<const DenseTensor *> &ins,
                          const phi::DDim &dims,
                          int axis) {
    N = std::max(static_cast<int>(ins.size()), 2);
    in_dims.resize(N);
    rank = dims.size();
    out_dims = phi::vectorize<int64_t>(dims);
    if (ins.size() == 1) {
      // When ins.size() = 1, broadcast input to output.
      in_dims[0] = phi::vectorize<int64_t>(ins[0]->dims());
      // Add out_dims to in_dims to avoid errors in dims merging.
      in_dims[1] = out_dims;
    } else {
      for (int j = 0; j < N; ++j) {
        in_dims[j] = phi::vectorize<int64_t>(ins[j]->dims());
      }
    }
    ExtendInputDimensions(N, axis);

    // To Merge the dimensions of input_tensors while the consequtive
    // equal-dimensions appears. Example below :
    //   in_1.shape = [2, 3, 4, 5]    in_1.shape = [2, 12, 5]
    //   in_2.shape = [1, 3, 4, 5] -> in_2.shape = [1, 12, 5]
    //   in_3.shape = [2, 3, 4, 1]    in_3.shape = [2, 12, 1]
    auto merge_sequential_dims = [](bool &equal,
                                    std::vector<DimVector> &in_dims,
                                    DimVector &out,
                                    int i,
                                    int num) {
      for (int j = 1; j < num; ++j) {
        equal &= (in_dims[0][i] == in_dims[j][i]) ? true : false;
      }
    };
    MergeFunctor merge_ptr = merge_sequential_dims;
    MergeDimensions<MergeFunctor>(merge_ptr, N);

    // To Merge the dimension of input_tensors while the sequential
    // 1-value-dimensions appears. Example below :
    //   in_1.shape = [2, 1, 1, 5]    in_1.shape = [2,  1, 5]
    //   in_2.shape = [2, 3, 4, 5] -> in_2.shape = [1, 12, 5]
    //   in_3.shape = [2, 3, 4, 1]    in_3.shape = [2, 12, 1]
    // Caution: Once 1-value-dimensions appears, the corresponding
    // shape position of other input tensors must be same with the
    // output tensor`s shape, or incorrect merge may occur.
    auto merge_sequential_one_dims = [](bool &equal,
                                        std::vector<DimVector> &in_dims,
                                        DimVector &out,
                                        int i,
                                        int num) {
      equal = in_dims[0][i] == 1;
      if (equal) {
        for (int j = 1; j < num; ++j) {
          equal &= in_dims[j][i] == out[i];
        }
      }
    };
    for (auto i = 0; i < rank; ++i) {
      int swap_idx = 0;
      bool has_seq_one = FindSequentialOneDim(&swap_idx);
      if (!has_seq_one) {
        break;
      }
      merge_ptr = merge_sequential_one_dims;
      MergeDimensions<MergeFunctor>(merge_ptr, N);
      std::swap(in_dims[swap_idx], in_dims[0]);
    }
  }

 private:
  // To compensate the lackage of input_tensors' dimension with axis.
  void ExtendInputDimensions(int N, int axis) {
    for (auto &in_dim : in_dims) {
      int64_t in_idx = 0;
      if (in_dim.size() < rank) {
        DimVector tmp_dim(rank, 1);
        for (; in_idx < in_dim.size();) {
          if (in_dim[in_idx] == out_dims[axis] || in_dim[in_idx] == 1) {
            tmp_dim[axis] = in_dim[in_idx];
            in_idx++;
            axis++;
          } else {
            PADDLE_THROW(phi::errors::InvalidArgument(
                "The %d-th dimension of input tensor is expected to be equal "
                "with the %d-th dimension of output tensor %d or 1, but "
                "received %d.",
                in_idx + 1,
                axis + 1,
                out_dims[axis],
                in_dim[in_idx]));
          }
        }
        in_dim.resize(rank);
        std::copy(tmp_dim.begin(), tmp_dim.end(), in_dim.begin());
      } else {
        for (; in_idx < rank;) {
          if (in_dim[in_idx] == out_dims[in_idx] || in_dim[in_idx] == 1) {
            in_idx++;
          } else {
            PADDLE_THROW(phi::errors::InvalidArgument(
                "The %d-th dimension of input tensor is expected to be equal "
                "with the %d-th dimension of output tensor %d or 1, but "
                "received %d.",
                in_idx + 1,
                in_idx + 1,
                out_dims[in_idx],
                in_dim[in_idx]));
          }
        }
      }
      std::reverse(in_dim.begin(), in_dim.end());
    }
    std::reverse(out_dims.begin(), out_dims.end());
  }

  // Merge sequential dimension to shrink calculation cost for
  // offset computation in CUDA Kernel.
  template <typename MergeFunctor>
  __inline__ void MergeDimensions(MergeFunctor merge_func, int N) {
    auto VectorReorganise = [](DimVector *vec, int l_idx, int m_idx) {
      (*vec)[m_idx - 1] = std::accumulate(vec->begin() + l_idx,
                                          vec->begin() + m_idx,
                                          1,
                                          std::multiplies<int64_t>());
      vec->erase(vec->begin() + l_idx, vec->begin() + m_idx - 1);
    };

    int64_t i = 0;
    while (i < rank) {
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
      } while (i < rank);

      if (cnt > 1) {
        for (auto &in_dim : in_dims) {
          VectorReorganise(&in_dim, low_idx, i);
        }
        VectorReorganise(&out_dims, low_idx, i);
        rank -= --cnt;
        i -= cnt;
      } else if (cnt < 1) {
        i++;
      }
    }
  }

  // To judge whether shape of any input tensors is sequential
  // 1-value-dimensions, and metric the length of it.
  bool FindSequentialOneDim(int *swap_index) {
    int index = 0;
    int max_one_length = 0;
    for (int j = 0; j < N; ++j) {
      int seq_one_length = 0;
      bool active_seq = false;

      for (int i = 0; i < rank; ++i) {
        if (!active_seq && in_dims[j][i] == 1) {
          seq_one_length = 1;
          active_seq = true;
        } else if (active_seq) {
          if (in_dims[j][i] == 1) {
            seq_one_length++;
          } else {
            active_seq = false;
          }
        }
      }
      index = seq_one_length > max_one_length ? j : index;
      max_one_length = std::max(seq_one_length, max_one_length);
    }

    bool has_seq_one = max_one_length > 1;
    if (has_seq_one) {
      std::swap(in_dims[0], in_dims[index]);
      *swap_index = index;
    }
    return has_seq_one;
  }
};

// Simplify the input dims and permute dims if possible.
struct PermuteDimsSimplifier {
 public:
  PermuteDimsSimplifier(const int rank,
                        const int64_t numel,
                        const std::vector<int32_t> &perm,
                        const std::vector<int64_t> &dims)
      : perm_(rank), src_dims_(rank), count_(numel) {
    SimplifyPermAndDims(rank, dims, perm);
    perm_.resize(rank_);
    src_dims_.resize(rank_);
    dst_dims_.resize(rank_);
    if (!is_sequential_perm_) {
      for (auto i = 0; i < rank_; ++i) {
        dst_dims_[i] = src_dims_[perm_[i]];
      }
    } else {
      dst_dims_[0] = numel;
      src_dims_[0] = numel;
    }
  }

  ~PermuteDimsSimplifier() = default;

  const int &GetRank() const { return rank_; }
  const int64_t &GetCount() const { return count_; }
  const std::vector<int> &GetPerm() const { return perm_; }
  const std::vector<int64_t> &GetSrcDims() const { return src_dims_; }
  const std::vector<int64_t> &GetDstDims() const { return dst_dims_; }

 private:
  int rank_{1};
  int64_t count_{0};
  std::vector<int> perm_;
  bool is_sequential_perm_{true};
  std::vector<int64_t> src_dims_;
  std::vector<int64_t> dst_dims_;

  void SimplifyPermAndDims(const int rank,
                           const std::vector<int64_t> &in_dims,
                           const std::vector<int32_t> &perm) {
    int start_perm_idx = 0;
    int valid_dim_idx = 0;
    int valid_map[phi::DDim::kMaxRank];
    int64_t combined_dims[phi::DDim::kMaxRank];

    // Merge consecutive dims to the fist one dim and
    // leave original dim to be 1. Example below :
    // perm: [2, 3, 0, 1], origin_dims : [4, 8, 2, 5]
    // new_dims: [4, 8, 2, 5] -> [32, 1, 10, 1]
    while (start_perm_idx < rank) {
      const int start_dim_idx = perm[start_perm_idx];
      combined_dims[start_dim_idx] = in_dims[start_dim_idx];
      int end_perm_idx = start_perm_idx + 1;

      while (end_perm_idx < rank &&
             perm[end_perm_idx] == perm[end_perm_idx - 1] + 1) {
        const int end_dim_idx = perm[end_perm_idx];
        combined_dims[start_dim_idx] *= in_dims[end_dim_idx];
        combined_dims[end_dim_idx] = 1;
        end_perm_idx += 1;
      }
      start_perm_idx = end_perm_idx;
    }

    // Reorder combined dims and marked useless dim as -1.
    // for example, if combined dims is [32, 1, 10, 1],
    // valid_map is [0, -1, 1, -1] and generate simplified
    // dims as [32, 10]
    for (auto i = 0; i < rank; ++i) {
      const int dim_val = combined_dims[i];
      if (dim_val == 1) {
        valid_map[i] = -1;
      } else {
        valid_map[i] = valid_dim_idx;
        src_dims_[valid_dim_idx] = dim_val;
        valid_dim_idx += 1;
      }
    }

    if (valid_dim_idx == 0) {
      src_dims_[0] = 1;
      perm_[0] = 0;
      return;
    }

    // Acquire simplified perm with help of combined dims
    // and original perm, finally simplified perm is [1, 0]
    int perm_idx = 0;
    for (auto i = 0; i < rank; ++i) {
      const int mapped = valid_map[perm[i]];
      if (mapped >= 0) {
        perm_[perm_idx] = mapped;
        is_sequential_perm_ &= (mapped == perm_idx);
        perm_idx += 1;
      }
    }
    rank_ = is_sequential_perm_ ? 1 : valid_dim_idx;
  }
};

template <typename T>
struct DimsSimplifiedLogger {
 public:
  static void Log(const std::vector<const DenseTensor *> &ins,
                  std::vector<DenseTensor *> *outs,
                  const BroadcastDimsSimplifier &dims_simplifier,
                  const std::string &op_name) {
    VLOG(6) << op_name << "`s dims after simplification is below :";
    for (size_t i = 0; i < ins.size(); ++i) {
      VLOG(6) << "input i=" << i << ": origin_dims={" << ins[i]->dims()
              << "}, simplied_dims={"
              << ReversedVectorToString(dims_simplifier.in_dims[i]) << "}";
    }
    VLOG(6) << "output: origin_dims={" << (*outs)[0]->dims()
            << "}, simplied_dims={"
            << ReversedVectorToString(dims_simplifier.out_dims) << "}";
  }

  static std::string ReversedVectorToString(const std::vector<T> &reversed_v) {
    std::stringstream ss;
    bool is_last = true;
    for (int i = reversed_v.size() - 1; i >= 0; --i) {
      if (is_last) {
        ss << reversed_v[i];
        is_last = false;
      } else {
        ss << ", " << reversed_v[i];
      }
    }
    return ss.str();
  }
};

}  // namespace funcs
}  // namespace phi
