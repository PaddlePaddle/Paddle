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

#include "paddle/phi/kernels/impl/lrn_kernel_impl.h"

#include <memory>
#include <string>
#include <vector>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#ifdef PADDLE_WITH_DNNL
#include "paddle/phi/backends/onednn/onednn_helper.h"
#endif

namespace phi {

template <typename T>
struct LRNGradFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& out,
                  const phi::DenseTensor& mid,
                  phi::DenseTensor* x_g,
                  const phi::DenseTensor& out_g,
                  int N,
                  int C,
                  int H,
                  int W,
                  int n,
                  T alpha,
                  T beta,
                  const DataLayout data_layout) {
    T ratio = -2 * alpha * beta;
    auto x_g_e = phi::EigenVector<T>::Flatten(*x_g);
    x_g_e = x_g_e.constant(0.0);

    auto e_x = phi::EigenTensor<T, 4>::From(x);
    auto e_x_g = phi::EigenTensor<T, 4>::From(*x_g);
    auto e_out = phi::EigenTensor<T, 4>::From(out);
    auto e_out_g = phi::EigenTensor<T, 4>::From(out_g);
    auto e_mid = phi::EigenTensor<T, 4>::From(mid);

    const int start = -(n - 1) / 2;
    const int end = start + n;
    for (int m = 0; m < N; m++) {
      for (int i = 0; i < C; i++) {
        auto offsets = Eigen::array<int, 4>({{m, i, 0, 0}});
        auto extents = Eigen::array<int, 4>({{1, 1, H, W}});
        if (data_layout == DataLayout::kNHWC) {
          offsets = Eigen::array<int, 4>({{m, 0, 0, i}});
          extents = Eigen::array<int, 4>({{1, H, W, 1}});
        }

        auto i_x = e_x.slice(offsets, extents);
        auto i_x_g = e_x_g.slice(offsets, extents);
        auto i_out_g = e_out_g.slice(offsets, extents);
        auto i_mid = e_mid.slice(offsets, extents);

        i_x_g = i_mid.pow(-beta) * i_out_g;
        for (int c = start; c < end; c++) {
          int ch = i + c;
          if (ch < 0 || ch >= C) {
            continue;
          }

          if (data_layout != DataLayout::kNHWC) {
            offsets = Eigen::array<int, 4>({{m, ch, 0, 0}});
          } else {
            offsets = Eigen::array<int, 4>({{m, 0, 0, ch}});
          }
          auto c_out = e_out.slice(offsets, extents);
          auto c_mid = e_mid.slice(offsets, extents);
          auto c_out_g = e_out_g.slice(offsets, extents);

          i_x_g += ratio * c_out_g * c_out * i_x / c_mid;
        }
      }
    }
  }
};
template struct LRNGradFunctor<phi::CPUContext, float>;
template struct LRNGradFunctor<phi::CPUContext, double>;
}  // namespace phi

PD_REGISTER_KERNEL(lrn_grad, CPU, ALL_LAYOUT, phi::LRNGradKernel, float) {}
