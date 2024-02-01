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

#include "paddle/phi/kernels/shuffle_batch_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename T, typename Context>
void ShuffleBatchKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& seed,
                        int startup_seed,
                        DenseTensor* out,
                        DenseTensor* shuffleidx,
                        DenseTensor* seed_out) {
  auto x_embed_size = x.dims()[x.dims().size() - 1];
  int elem_size = 1;
  for (auto i = 0; i < x.dims().size() - 1; i++)
    elem_size *= static_cast<int>(x.dims()[i]);

  std::vector<int64_t> idx_vec;  // record shuffled order
  idx_vec.reserve(elem_size);
  for (int i = 0; i < elem_size; i++) {
    idx_vec.push_back(i);
  }
  int64_t seed_int = 0;
  if (seed.initialized()) {
    seed_int = *seed.data<int64_t>();
  } else {
    seed_int = startup_seed;
  }
  std::default_random_engine engine;
  engine.seed(seed_int);

  auto custom_random_shuffle = [&idx_vec]() {
    std::random_device rnd;
    int64_t seed_tmp = rnd();
    std::default_random_engine rng(seed_tmp);
    const int n = static_cast<int>(idx_vec.size());
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    std::vector<bool> visit(n, false);
    while (!v.empty()) {
      std::shuffle(v.begin(), v.end(), rng);
      int tmp = v.back();
      v.pop_back();
      if (v.empty()) {
        std::uniform_int_distribution<int> distr(0, n - 2);
        idx_vec[tmp] = tmp;
        std::swap(idx_vec[tmp], idx_vec[(distr(rng) + tmp + 1) % n]);
        return;
      }
      visit[tmp] = true;
      std::shuffle(v.begin(), v.end(), rng);
      int curr = v.back();
      v.pop_back();
      v.push_back(tmp);
      idx_vec[tmp] = curr;
      while (!visit[curr]) {
        visit[curr] = true;
        std::shuffle(v.begin(), v.end(), rng);
        idx_vec[curr] = v.back();
        v.pop_back();
        curr = static_cast<int>(idx_vec[curr]);
      }
    }
  };
  custom_random_shuffle();
  // change shuffle to custom_random_shuffle
  // std::shuffle(idx_vec.begin(), idx_vec.end(), engine);

  // ShuffleIdx record shuffle order
  shuffleidx->Resize(common::make_ddim({(int64_t)idx_vec.size()}));
  auto* shuffleidx_data = dev_ctx.template HostAlloc<int64_t>(shuffleidx);

  for (size_t i = 0; i < idx_vec.size(); i++) {
    shuffleidx_data[i] = idx_vec[i];
  }
  // copy data according to idx_vec
  auto* x_data = x.data<T>();
  auto* out_data = dev_ctx.template HostAlloc<T>(out);

  for (auto i = 0; i < elem_size; i++) {
    memcpy(out_data + idx_vec[i] * x_embed_size,
           x_data + i * x_embed_size,
           x_embed_size * sizeof(T));
  }
  // set new seed
  seed_out->Resize(common::make_ddim({1}));
  auto* seed_out_data = dev_ctx.template HostAlloc<int64_t>(seed_out);
  *seed_out_data = engine();
}
}  // namespace phi

PD_REGISTER_KERNEL(shuffle_batch,
                   CPU,
                   ALL_LAYOUT,
                   phi::ShuffleBatchKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT64);
}
