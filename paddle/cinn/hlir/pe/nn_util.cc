// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/pe/nn_util.h"

#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/common/target.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace hlir {
namespace pe {

using cinn::lang::Compute;
using ir::Tensor;

std::vector<std::vector<std::vector<float>>> get_winograd_val(
    const int& tile_size, const int& kernel_size) {
  std::unordered_map<std::string, std::vector<std::vector<std::vector<float>>>>
      all_vals;
  {
    std::string keys = "2+3";
    std::vector<std::vector<std::vector<float>>> nums;
    std::vector<std::vector<float>> A = {
        {1., 0.}, {1., -1.}, {1., 1.}, {0., 1.}};
    nums.push_back(A);
    std::vector<std::vector<float>> B = {{1., 0., 0., 0.},
                                         {0., -1., 1., -1.},
                                         {-1., 1., 1., 0.},
                                         {0., 0., 0., 1.}};
    nums.push_back(B);
    std::vector<std::vector<float>> G = {
        {1., 0., 0.}, {0.5, -0.5, 0.5}, {0.5, 0.5, 0.5}, {0., 0., 1.}};
    nums.push_back(G);
    all_vals[keys] = nums;
  }
  {
    std::string keys = "2+5";
    std::vector<std::vector<std::vector<float>>> nums;
    std::vector<std::vector<float>> A = {{1.0, 0.0},
                                         {1.0, -1.0},
                                         {1.0, 1.0},
                                         {1.0, 0.5},
                                         {1.0, -2.0},
                                         {0.0, 1.0}};
    nums.push_back(A);
    std::vector<std::vector<float>> B = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                         {-1.5, 1.0, -1.0, -2.0, 0.5, 1.0},
                                         {-2.0, -2.5, 0.5, -1.0, -1.0, -1.5},
                                         {1.5, 0.5, 2.5, 2.0, -0.5, -2.0},
                                         {1.0, 1.0, 1.0, 1.0, 1.0, 1.5},
                                         {0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    nums.push_back(B);
    std::vector<std::vector<float>> G = {{1.0, 0.0, 0.0, 0.0, 0.0},
                                         {-0.3333333333333333,
                                          0.3333333333333333,
                                          -0.3333333333333333,
                                          0.3333333333333333,
                                          -0.3333333333333333},
                                         {0.3333333333333333,
                                          0.3333333333333333,
                                          0.3333333333333333,
                                          0.3333333333333333,
                                          0.3333333333333333},
                                         {-1.0666666666666667,
                                          -0.5333333333333333,
                                          -0.26666666666666666,
                                          -0.13333333333333333,
                                          -0.06666666666666667},
                                         {0.06666666666666667,
                                          -0.13333333333333333,
                                          0.26666666666666666,
                                          -0.5333333333333333,
                                          1.0666666666666667},
                                         {0.0, 0.0, 0.0, 0.0, 1.0}};
    nums.push_back(G);
    all_vals[keys] = nums;
  }
  {
    std::string keys = "2+7";
    std::vector<std::vector<std::vector<float>>> nums;
    std::vector<std::vector<float>> A = {{1.0, 0.0},
                                         {1.0, -1.0},
                                         {1.0, 1.0},
                                         {1.0, 0.5},
                                         {1.0, -0.5},
                                         {1.0, 2.0},
                                         {1.0, -2.0},
                                         {0.0, 1.0}};
    nums.push_back(A);
    std::vector<std::vector<float>> B = {
        {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, -1.0, 1.0, 2.0, -2.0, 0.5, -0.5, -1.0},
        {-5.25, 1.0, 1.0, 4.0, 4.0, 0.25, 0.25, 0.0},
        {0.0, 4.25, -4.25, -2.5, 2.5, -2.5, 2.5, 5.25},
        {5.25, -4.25, -4.25, -5.0, -5.0, -1.25, -1.25, 0.0},
        {0.0, -1.0, 1.0, 0.5, -0.5, 2.0, -2.0, -5.25},
        {-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    nums.push_back(B);
    std::vector<std::vector<float>> G = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                         {-0.2222222222222222,
                                          0.2222222222222222,
                                          -0.2222222222222222,
                                          0.2222222222222222,
                                          -0.2222222222222222,
                                          0.2222222222222222,
                                          -0.2222222222222222},
                                         {-0.2222222222222222,
                                          -0.2222222222222222,
                                          -0.2222222222222222,
                                          -0.2222222222222222,
                                          -0.2222222222222222,
                                          -0.2222222222222222,
                                          -0.2222222222222222},
                                         {0.7111111111111111,
                                          0.35555555555555557,
                                          0.17777777777777778,
                                          0.08888888888888889,
                                          0.044444444444444446,
                                          0.022222222222222223,
                                          0.011111111111111112},
                                         {0.7111111111111111,
                                          -0.35555555555555557,
                                          0.17777777777777778,
                                          -0.08888888888888889,
                                          0.044444444444444446,
                                          -0.022222222222222223,
                                          0.011111111111111112},
                                         {0.011111111111111112,
                                          0.022222222222222223,
                                          0.044444444444444446,
                                          0.08888888888888889,
                                          0.17777777777777778,
                                          0.35555555555555557,
                                          0.7111111111111111},
                                         {0.011111111111111112,
                                          -0.022222222222222223,
                                          0.044444444444444446,
                                          -0.08888888888888889,
                                          0.17777777777777778,
                                          -0.35555555555555557,
                                          0.7111111111111111},
                                         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    nums.push_back(G);
    all_vals[keys] = nums;
  }
  {
    std::string keys = "4+3";
    std::vector<std::vector<std::vector<float>>> nums;
    std::vector<std::vector<float>> A = {{1.0, 0.0, 0.0, 0.0},
                                         {1.0, -1.0, 1.0, -1.0},
                                         {1.0, 1.0, 1.0, 1.0},
                                         {1.0, 0.5, 0.25, 0.125},
                                         {1.0, -2.0, 4.0, -8.0},
                                         {0.0, 0.0, 0.0, 1.0}};
    nums.push_back(A);
    std::vector<std::vector<float>> B = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                         {-1.5, 1.0, -1.0, -2.0, 0.5, 1.0},
                                         {-2.0, -2.5, 0.5, -1.0, -1.0, -1.5},
                                         {1.5, 0.5, 2.5, 2.0, -0.5, -2.0},
                                         {1.0, 1.0, 1.0, 1.0, 1.0, 1.5},
                                         {0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    nums.push_back(B);
    std::vector<std::vector<float>> G = {
        {1.0, 0.0, 0.0},
        {-0.3333333333333333, 0.3333333333333333, -0.3333333333333333},
        {0.3333333333333333, 0.3333333333333333, 0.3333333333333333},
        {-1.0666666666666667, -0.5333333333333333, -0.26666666666666666},
        {0.06666666666666667, -0.13333333333333333, 0.26666666666666666},
        {0.0, 0.0, 1.0}};
    nums.push_back(G);
    all_vals[keys] = nums;
  }
  {
    std::string keys = "4+5";
    std::vector<std::vector<std::vector<float>>> nums;
    std::vector<std::vector<float>> A = {{1.0, 0.0, 0.0, 0.0},
                                         {1.0, -1.0, 1.0, -1.0},
                                         {1.0, 1.0, 1.0, 1.0},
                                         {1.0, 0.5, 0.25, 0.125},
                                         {1.0, -0.5, 0.25, -0.125},
                                         {1.0, 2.0, 4.0, 8.0},
                                         {1.0, -2.0, 4.0, -8.0},
                                         {0.0, 0.0, 0.0, 1.0}};
    nums.push_back(A);
    std::vector<std::vector<float>> B = {
        {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, -1.0, 1.0, 2.0, -2.0, 0.5, -0.5, -1.0},
        {-5.25, 1.0, 1.0, 4.0, 4.0, 0.25, 0.25, 0.0},
        {0.0, 4.25, -4.25, -2.5, 2.5, -2.5, 2.5, 5.25},
        {5.25, -4.25, -4.25, -5.0, -5.0, -1.25, -1.25, 0.0},
        {0.0, -1.0, 1.0, 0.5, -0.5, 2.0, -2.0, -5.25},
        {-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    nums.push_back(B);
    std::vector<std::vector<float>> G = {{1.0, 0.0, 0.0, 0.0, 0.0},
                                         {-0.2222222222222222,
                                          0.2222222222222222,
                                          -0.2222222222222222,
                                          0.2222222222222222,
                                          -0.2222222222222222},
                                         {-0.2222222222222222,
                                          -0.2222222222222222,
                                          -0.2222222222222222,
                                          -0.2222222222222222,
                                          -0.2222222222222222},
                                         {0.7111111111111111,
                                          0.35555555555555557,
                                          0.17777777777777778,
                                          0.08888888888888889,
                                          0.044444444444444446},
                                         {0.7111111111111111,
                                          -0.35555555555555557,
                                          0.17777777777777778,
                                          -0.08888888888888889,
                                          0.044444444444444446},
                                         {0.011111111111111112,
                                          0.022222222222222223,
                                          0.044444444444444446,
                                          0.08888888888888889,
                                          0.17777777777777778},
                                         {0.011111111111111112,
                                          -0.022222222222222223,
                                          0.044444444444444446,
                                          -0.08888888888888889,
                                          0.17777777777777778},
                                         {0.0, 0.0, 0.0, 0.0, 1.0}};
    nums.push_back(G);
    all_vals[keys] = nums;
  }
  {
    std::string keys = "4+7";
    std::vector<std::vector<std::vector<float>>> nums;
    std::vector<std::vector<float>> A = {{1.0, 0.0, 0.0, 0.0},
                                         {1.0, -1.0, 1.0, -1.0},
                                         {1.0, 1.0, 1.0, 1.0},
                                         {1.0, 0.5, 0.25, 0.125},
                                         {1.0, -0.5, 0.25, -0.125},
                                         {1.0, 2.0, 4.0, 8.0},
                                         {1.0, -2.0, 4.0, -8.0},
                                         {1.0, -0.25, 0.0625, -0.015625},
                                         {1.0, 4.0, 16.0, 64.0},
                                         {0.0, 0.0, 0.0, 1.0}};
    nums.push_back(A);
    std::vector<std::vector<float>> B = {
        {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.75, 1.0, -1.0, -2.0, 2.0, -0.5, 0.5, 4.0, -0.25, 1.0},
        {-6.25,
         2.75,
         -4.75,
         -11.5,
         3.4999999999999996,
         -2.125,
         1.625,
         -1.0,
         -1.0,
         3.75},
        {-19.6875,
         -9.0,
         1.5,
         -10.5,
         -19.5,
         2.0625,
         -3.9375,
         -21.0,
         1.3125,
         -6.25},
        {10.5,
         -10.6875,
         21.1875,
         18.375,
         -0.375,
         10.875,
         -7.875,
         5.25,
         5.25,
         -19.6875},
        {19.6875,
         21.1875,
         10.6875,
         15.75,
         21.75,
         0.1875,
         9.1875,
         21.0,
         -1.3125,
         10.5},
        {-6.25, -1.5, -9.0, -7.875, -4.125, -9.75, 5.25, -5.25, -5.25, 19.6875},
        {-3.75,
         -4.75,
         -2.75,
         -3.25,
         -4.25,
         -1.7499999999999998,
         -5.75,
         -4.0,
         0.25,
         -6.25},
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -3.7500000000000004},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    nums.push_back(B);
    std::vector<std::vector<float>> G = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                         {-0.05925925925925926,
                                          0.05925925925925926,
                                          -0.05925925925925926,
                                          0.05925925925925926,
                                          -0.05925925925925926,
                                          0.05925925925925926,
                                          -0.05925925925925926},
                                         {0.05925925925925926,
                                          0.05925925925925926,
                                          0.05925925925925926,
                                          0.05925925925925926,
                                          0.05925925925925926,
                                          0.05925925925925926,
                                          0.05925925925925926},
                                         {-0.2708994708994709,
                                          -0.13544973544973546,
                                          -0.06772486772486773,
                                          -0.033862433862433865,
                                          -0.016931216931216932,
                                          -0.008465608465608466,
                                          -0.004232804232804233},
                                         {0.6320987654320988,
                                          -0.3160493827160494,
                                          0.1580246913580247,
                                          -0.07901234567901234,
                                          0.03950617283950617,
                                          -0.019753086419753086,
                                          0.009876543209876543},
                                         {-0.0024691358024691358,
                                          -0.0049382716049382715,
                                          -0.009876543209876543,
                                          -0.019753086419753086,
                                          -0.03950617283950617,
                                          -0.07901234567901234,
                                          -0.1580246913580247},
                                         {0.0010582010582010583,
                                          -0.0021164021164021165,
                                          0.004232804232804233,
                                          -0.008465608465608466,
                                          0.016931216931216932,
                                          -0.033862433862433865,
                                          0.06772486772486773},
                                         {-1.3598091088287168,
                                          0.3399522772071792,
                                          -0.0849880693017948,
                                          0.0212470173254487,
                                          -0.005311754331362175,
                                          0.0013279385828405437,
                                          -0.0003319846457101359},
                                         {2.0749040356883495e-05,
                                          8.299616142753398e-05,
                                          0.0003319846457101359,
                                          0.0013279385828405437,
                                          0.005311754331362175,
                                          0.0212470173254487,
                                          0.0849880693017948},
                                         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    nums.push_back(G);
    all_vals[keys] = nums;
  }
  std::string keys =
      std::to_string(tile_size) + "+" + std::to_string(kernel_size);
  return all_vals[keys];
}

ir::Tensor const_matrix(const std::vector<std::vector<float>>& input,
                        const std::string& name) {
  int row = input.size();
  int col = input[0].size();
  std::vector<Expr> tensor_shape = {Expr(row), Expr(col)};
  auto result = Compute(
      tensor_shape,
      [=](Expr yy, Expr xx) {
        auto now = cinn::common::make_const(1.0f);
        for (int ii = 0; ii < row; ii++) {
          for (int jj = 0; jj < col; jj++) {
            // if (cinn::common::is_zero(Expr(ii)-yy) &&
            // cinn::common::is_zero(Expr(jj)-xx))
            // {
            //     now = cinn::common::make_const(input[ii][jj]);
            // }
            auto cond =
                cinn::common::and_all({Expr(ii) - yy == 0, Expr(jj) - xx == 0});
            now = cinn::common::select(
                cond, cinn::common::make_const(input[ii][jj]), now);
          }
        }
        return now;
      },
      name);
  return result;
}

std::vector<ir::Tensor> winograd_transform_matrices(const int& tile_size,
                                                    const int& kernel_size) {
  std::vector<std::vector<std::vector<float>>> vals =
      get_winograd_val(tile_size, kernel_size);
  PADDLE_ENFORCE_EQ(vals.size(),
                    3U,
                    ::common::errors::InvalidArgument(
                        "vals_size of winograd is not 3! Please check."));

  std::vector<std::vector<float>> A = vals[0];
  std::vector<std::vector<float>> B = vals[1];
  std::vector<std::vector<float>> G = vals[2];

  std::string name_a = "A_matrix";
  auto tensor_a = const_matrix(A, name_a);

  std::string name_b = "B_matrix";
  auto tensor_b = const_matrix(B, name_b);

  std::string name_g = "G_matrix";
  auto tensor_g = const_matrix(G, name_g);

  return {tensor_a, tensor_b, tensor_g};
}

int GetPostParallelSize(const std::vector<int>& inshape,
                        const std::vector<int>& axes) {
  int parallel_size = 1;
  for (int idx = axes.back() + 1; idx < inshape.size(); ++idx) {
    parallel_size *= inshape[idx];
  }
  return parallel_size;
}

int GetParallelSize(const std::vector<int>& inshape,
                    const std::vector<int>& axes) {
  int parallel_size = 1;
  for (int idx = 0; idx < inshape.size(); ++idx) {
    if (std::find(axes.begin(), axes.end(), idx) != axes.end()) {
      continue;
    }
    parallel_size *= inshape[idx];
  }
  return parallel_size;
}

int GetTailSize(const std::vector<int>& inshape, const std::vector<int>& axes) {
  int tail_size = 1;
  for (int idx = axes.back() + 1; idx < inshape.size(); ++idx) {
    tail_size *= inshape[idx];
  }
  return tail_size;
}

std::vector<int> GetFirstStepReduceShape(const std::vector<int>& shape,
                                         const std::vector<int>& axes,
                                         bool& inbound,  // NOLINT
                                         int& tail) {    // NOLINT
  // post parallel size
  int post_parallel_size = GetPostParallelSize(shape, axes);
  // the size to unfold las reduce axis
  int unfold_size =
      cinn::common::GetMaxThreads() / GetParallelSize(shape, axes);
  PADDLE_ENFORCE_GT(unfold_size,
                    1,
                    ::common::errors::InvalidArgument(
                        "unfold_size should be greater than 1!"));

  // fuse reduce axis.
  int insert_zero_num = 0;
  int last_axis_index = axes.size() - 1;
  int last_reduce_size = shape[axes.back()];
  for (; last_axis_index >= 1; --last_axis_index) {
    if (axes[last_axis_index] - 1 != axes[last_axis_index - 1]) {
      break;
    }
    ++insert_zero_num;
    int index = axes[last_axis_index - 1];
    last_reduce_size *= shape[index];
  }

  std::vector<int> reduce_shape;
  for (int idx = 0; idx < axes[last_axis_index]; ++idx) {
    reduce_shape.push_back(shape[idx]);
  }

  // insert 1 to keep dimension size.
  for (int idx = 0; idx < insert_zero_num; ++idx) {
    reduce_shape.emplace_back(1);
  }

  // get tail size.
  if (last_reduce_size < unfold_size && last_reduce_size < 64) {
    reduce_shape.emplace_back(1);
    reduce_shape.emplace_back(last_reduce_size);

    for (int idx = axes.back() + 1; idx < shape.size(); ++idx) {
      reduce_shape.push_back(shape[idx]);
    }
    return reduce_shape;
  }

  // set loop size set.
  static std::vector<int> loop_size_set = {64, 48, 32, 24, 16, 12, 8, 4, 2, 1};
  for (auto loop_size : loop_size_set) {
    if (last_reduce_size < loop_size || unfold_size < loop_size) {
      continue;
    }

    if (last_reduce_size % loop_size != 0) {
      continue;
    }

    if (loop_size > 1) {
      reduce_shape.emplace_back(last_reduce_size / loop_size);
      reduce_shape.emplace_back(loop_size);
    } else {
      for (auto loop_size : loop_size_set) {
        if (unfold_size < loop_size) {
          continue;
        }
        tail = last_reduce_size - (last_reduce_size / loop_size) * loop_size;
        // do ceil
        reduce_shape.emplace_back((last_reduce_size - 1) / loop_size + 1);
        reduce_shape.emplace_back(loop_size);
        inbound = false;
        break;
      }
    }
    break;
  }

  // std::vector<ir::Expr> tail;
  for (int idx = axes.back() + 1; idx < shape.size(); ++idx) {
    reduce_shape.push_back(shape[idx]);
  }

  return reduce_shape;
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
