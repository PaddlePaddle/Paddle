// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <class T>
bool DistPairDescend(std::tuple<int, int, T> pair1,
                     std::tuple<int, int, T> pair2) {
  return std::get<2>(pair1) > std::get<2>(pair2);
}

// The match_indices must be initialized to -1 at first.
// The match_dist must be initialized to 0 at first.
template <typename T>
void BipartiteMatch(const phi::DenseTensor& dist,
                    int* match_indices,
                    T* match_dist) {
  PADDLE_ENFORCE_EQ(
      dist.dims().size(),
      2,
      common::errors::InvalidArgument("The rank of dist must be 2."));
  int64_t row = dist.dims()[0];
  int64_t col = dist.dims()[1];
  auto* dist_data = dist.data<T>();
  // Test result: When row==130 the speed of these two methods almost the same
  if (row >= 130) {
    std::vector<std::tuple<int, int, T>> match_pair;

    for (int64_t i = 0; i < row; ++i) {
      for (int64_t j = 0; j < col; ++j) {
        match_pair.push_back(std::make_tuple(i, j, dist_data[i * col + j]));
      }
    }
    std::sort(match_pair.begin(), match_pair.end(), DistPairDescend<T>);
    std::vector<int> row_indices(row, -1);

    int64_t idx = 0;
    for (int64_t k = 0; k < row * col; ++k) {
      int64_t i = std::get<0>(match_pair[k]);
      int64_t j = std::get<1>(match_pair[k]);
      T dist = std::get<2>(match_pair[k]);

      if (idx >= row) {
        break;
      }
      if (match_indices[j] == -1 && row_indices[i] == -1 && dist > 0) {
        match_indices[j] = static_cast<int>(i);
        row_indices[i] = static_cast<int>(j);
        match_dist[j] = dist;
        idx += 1;
      }
    }
  } else {
    constexpr T kEPS = static_cast<T>(1e-6);
    std::vector<int> row_pool;
    for (int i = 0; i < row; ++i) {
      row_pool.push_back(i);
    }
    while (!row_pool.empty()) {
      int max_idx = -1;
      int max_row_idx = -1;
      T max_dist = -1;
      for (int64_t j = 0; j < col; ++j) {
        if (match_indices[j] != -1) {
          continue;
        }
        for (auto m : row_pool) {
          // distance is 0 between m-th row and j-th column
          if (dist_data[m * col + j] < kEPS) {
            continue;
          }
          if (dist_data[m * col + j] > max_dist) {
            max_idx = static_cast<int>(j);
            max_row_idx = m;
            max_dist = dist_data[m * col + j];
          }
        }
      }
      if (max_idx == -1) {
        // Cannot find good match.
        break;
      } else {
        PADDLE_ENFORCE_EQ(
            match_indices[max_idx],
            -1,
            common::errors::InvalidArgument(
                "The match_indices must be initialized to -1 at [%d].",
                max_idx));
        match_indices[max_idx] = max_row_idx;
        match_dist[max_idx] = max_dist;
        // Erase the row index.
        row_pool.erase(
            std::find(row_pool.begin(), row_pool.end(), max_row_idx));
      }
    }
  }
}

template <typename T>
void ArgMaxMatch(const phi::DenseTensor& dist,
                 int* match_indices,
                 T* match_dist,
                 T overlap_threshold) {
  constexpr T kEPS = static_cast<T>(1e-6);
  int64_t row = dist.dims()[0];
  int64_t col = dist.dims()[1];
  auto* dist_data = dist.data<T>();
  for (int64_t j = 0; j < col; ++j) {
    if (match_indices[j] != -1) {
      // the j-th column has been matched to one entity.
      continue;
    }
    int max_row_idx = -1;
    T max_dist = -1;
    for (int i = 0; i < row; ++i) {
      T dist = dist_data[i * col + j];
      if (dist < kEPS) {
        // distance is 0 between m-th row and j-th column
        continue;
      }
      if (dist >= overlap_threshold && dist > max_dist) {
        max_row_idx = i;
        max_dist = dist;
      }
    }
    if (max_row_idx != -1) {
      PADDLE_ENFORCE_EQ(
          match_indices[j],
          -1,
          common::errors::InvalidArgument(
              "The match_indices must be initialized to -1 at [%d].", j));
      match_indices[j] = max_row_idx;
      match_dist[j] = max_dist;
    }
  }
}

template <typename T, typename Context>
void BipartiteMatchKernel(const Context& dev_ctx,
                          const DenseTensor& dist_mat_in,
                          const std::string& match_type,
                          float dist_threshold,
                          DenseTensor* col_to_row_match_indices,
                          DenseTensor* col_to_row_match_dist) {
  auto* dist_mat = &dist_mat_in;
  auto* match_indices = col_to_row_match_indices;
  auto* match_dist = col_to_row_match_dist;

  auto col = dist_mat->dims()[1];

  int64_t n = dist_mat->lod().empty()
                  ? 1
                  : static_cast<int64_t>(dist_mat->lod().back().size() - 1);
  if (!dist_mat->lod().empty()) {
    PADDLE_ENFORCE_EQ(
        dist_mat->lod().size(),
        1UL,
        common::errors::InvalidArgument("Only support 1 level of LoD."));
  }
  match_indices->Resize({n, col});
  dev_ctx.template Alloc<int>(match_indices);
  match_dist->Resize({n, col});
  dev_ctx.template Alloc<T>(match_dist);

  phi::funcs::SetConstant<phi::CPUContext, int> iset;
  iset(dev_ctx, match_indices, static_cast<int>(-1));
  phi::funcs::SetConstant<phi::CPUContext, T> tset;
  tset(dev_ctx, match_dist, static_cast<T>(0));

  int* indices = match_indices->data<int>();
  T* dist = match_dist->data<T>();
  auto type = match_type;
  auto threshold = dist_threshold;
  if (n == 1) {
    BipartiteMatch<T>(*dist_mat, indices, dist);
    if (type == "per_prediction") {
      ArgMaxMatch<T>(*dist_mat, indices, dist, threshold);
    }
  } else {
    auto lod = dist_mat->lod().back();
    for (size_t i = 0; i < lod.size() - 1; ++i) {
      if (lod[i + 1] > lod[i]) {
        phi::DenseTensor one_ins = dist_mat->Slice(
            static_cast<int64_t>(lod[i]), static_cast<int64_t>(lod[i + 1]));
        BipartiteMatch<T>(one_ins, indices + i * col, dist + i * col);
        if (type == "per_prediction") {
          ArgMaxMatch<T>(one_ins, indices + i * col, dist + i * col, threshold);
        }
      }
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(bipartite_match,
                   CPU,
                   ALL_LAYOUT,
                   phi::BipartiteMatchKernel,
                   float,
                   double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
}
