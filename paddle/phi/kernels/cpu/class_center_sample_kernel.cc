//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <set>
#include <vector>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ClassCenterSampleKernel(const Context& dev_ctx,
                             const DenseTensor& label,
                             int num_classes,
                             int num_samples,
                             int ring_id,
                             int rank,
                             int nranks,
                             bool fix_seed,
                             int seed,
                             DenseTensor* remapped_label,
                             DenseTensor* sampled_local_class_center) {
  PADDLE_ENFORCE_GT(num_classes,
                    0,
                    errors::InvalidArgument(
                        "The value 'num_classes' for Op(class_center_sample) "
                        "must be greater than 0, "
                        "but the value given is %d.",
                        num_classes));

  PADDLE_ENFORCE_GT(num_samples,
                    0,
                    errors::InvalidArgument(
                        "The value 'num_samples' for Op(class_center_sample) "
                        "must be greater than 0, "
                        "but the value given is %d.",
                        num_samples));

  PADDLE_ENFORCE_LE(num_samples,
                    num_classes,
                    errors::InvalidArgument(
                        "The value 'num_samples' for Op(class_center_sample) "
                        "must be less than or equal to %d, "
                        "but the value given is %d.",
                        num_classes,
                        num_samples));

  int64_t numel = label.numel();
  auto* label_ptr = label.data<T>();

  // get unique positive class center by ascending
  std::set<T, std::less<T>> unique_label;
  for (int64_t i = 0; i < numel; ++i) {
    unique_label.insert(label_ptr[i]);
  }

  // constrcut a lookup table and get sampled_local_class_center
  std::vector<T> actual_sampled;
  std::map<T, T> new_class_dict;
  T idx = 0;
  for (auto& t : unique_label) {
    new_class_dict[t] = idx;
    actual_sampled.push_back(t);
    idx++;
  }

  if (!fix_seed) {
    std::random_device rnd;
    seed = rnd();
  }
  std::uniform_int_distribution<T> dist(0, num_classes - 1);
  std::shared_ptr<std::mt19937_64> engine;
  if (seed) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);
  } else {
    engine = dev_ctx.GetGenerator()->GetCPUEngine();
  }
  // sample negative class center randomly
  while (unique_label.size() < static_cast<size_t>(num_samples)) {
    T neg = dist(*engine);
    if (unique_label.find(neg) == unique_label.end()) {
      unique_label.insert(neg);
      // unorder for negative class center
      actual_sampled.push_back(neg);
    }
  }

  int actual_num_samples = unique_label.size();
  sampled_local_class_center->Resize({actual_num_samples});
  T* sampled_local_class_center_ptr =
      dev_ctx.template Alloc<T>(sampled_local_class_center);

  idx = 0;
  for (auto& t : actual_sampled) {
    sampled_local_class_center_ptr[idx] = t;
    idx++;
  }

  // remap the input label to sampled class
  auto* remmaped_label_ptr = dev_ctx.template Alloc<T>(remapped_label);
  for (int64_t i = 0; i < numel; ++i) {
    remmaped_label_ptr[i] = new_class_dict[label_ptr[i]];
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(class_center_sample,
                   CPU,
                   ALL_LAYOUT,
                   phi::ClassCenterSampleKernel,
                   int64_t,
                   int) {}
