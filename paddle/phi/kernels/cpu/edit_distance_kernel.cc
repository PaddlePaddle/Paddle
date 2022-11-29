// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/edit_distance_kernel.h"

#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void EditDistanceKernel(const Context& ctx,
                        const DenseTensor& hyps,
                        const DenseTensor& refs,
                        const paddle::optional<DenseTensor>& hypslength,
                        const paddle::optional<DenseTensor>& refslength,
                        bool normalized,
                        DenseTensor* sequencenum,
                        DenseTensor* out) {
  int64_t* seq_num_data = ctx.template Alloc<int64_t>(sequencenum);
  auto batch_size = hyps.dims()[0];

  paddle::framework::Vector<size_t> hyp_lod(batch_size + 1);
  paddle::framework::Vector<size_t> ref_lod(batch_size + 1);

  bool use_length = hypslength.get_ptr() != nullptr;

  if (use_length) {
    // build lod when using padding
    auto hyp_length_ptr = hypslength.get_ptr()->data<int64_t>();
    auto ref_length_ptr = refslength.get_ptr()->data<int64_t>();

    for (auto i = 0; i < batch_size; i++) {
      hyp_lod[i + 1] = hyp_lod[i] + hyp_length_ptr[i];
      ref_lod[i + 1] = ref_lod[i] + ref_length_ptr[i];
    }

  } else {
    hyp_lod = hyps.lod()[0];
    ref_lod = refs.lod()[0];
  }

  if (normalized) {
    for (size_t i = 1; i < ref_lod.size(); ++i) {
      PADDLE_ENFORCE_GT(
          ref_lod[i],
          ref_lod[i - 1],
          errors::InvalidArgument("Reference string %d is empty.", i));
    }
  }
  auto num_strs = hyp_lod.size() - 1;
  *seq_num_data = static_cast<int64_t>(num_strs);

  out->Resize({static_cast<int64_t>(num_strs), 1});
  ctx.template Alloc<T>(out);
  auto outdata = out->data<T>();

  T distance = 0.0;
  for (size_t num = 0; num < num_strs; ++num) {
    auto m = static_cast<int64_t>(hyp_lod[num + 1] - hyp_lod[num]);
    auto n = static_cast<int64_t>(ref_lod[num + 1] - ref_lod[num]);

    if (m == 0) {
      distance = n;
    } else if (n == 0) {
      distance = m;
    } else {
      DenseTensor dist_t;
      dist_t.Resize({m + 1, n + 1});
      ctx.template Alloc<T>(&dist_t);
      auto dist = dist_t.data<T>();
      auto hyp_offset = use_length ? num * hyps.dims()[1] : hyp_lod[num];
      auto ref_offset = use_length ? num * refs.dims()[1] : ref_lod[num];
      auto x1 = hyps.data<int64_t>() + hyp_offset;
      auto x2 = refs.data<int64_t>() + ref_offset;
      for (int64_t i = 0; i < m + 1; ++i) {
        dist[i * (n + 1)] = i;
      }
      for (int64_t j = 0; j < n + 1; ++j) {
        dist[j] = j;
      }
      for (int64_t i = 1; i < m + 1; ++i) {
        for (int64_t j = 1; j < n + 1; ++j) {
          int cost = x1[i - 1] == x2[j - 1] ? 0 : 1;
          int dels = dist[(i - 1) * (n + 1) + j] + 1;
          int ins = dist[i * (n + 1) + (j - 1)] + 1;
          int subs = dist[(i - 1) * (n + 1) + (j - 1)] + cost;
          dist[i * (n + 1) + j] = std::min(dels, std::min(ins, subs));
        }
      }
      distance = dist[m * (n + 1) + n];
    }

    if (normalized) {
      PADDLE_ENFORCE_GT(
          n,
          0UL,
          errors::InvalidArgument("The reference string (#%d) cannot be empty "
                                  "when Attr(normalized) is enabled.",
                                  n));
      distance = distance / n;
    }
    outdata[num] = distance;
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    edit_distance, CPU, ALL_LAYOUT, phi::EditDistanceKernel, float) {}
