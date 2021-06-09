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

#pragma once
#include <map>
#include <set>
#include <vector>
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
template <typename T>
class ClassCenterSampleCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    auto* sampled_class = context.Output<LoDTensor>("SampledClass");
    int num_class = context.Attr<int>("num_class");
    float ratio = context.Attr<float>("ratio");
    int ignore_label = context.Attr<int>("ignore_label");
    unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));
    PADDLE_ENFORCE_GT(num_class, 0,
                      platform::errors::InvalidArgument(
                          "The value 'num_class' for Op(class_center_sample) "
                          "must be greater than 0, "
                          "but the value given is %d.",
                          num_class));
    PADDLE_ENFORCE_GT(ratio, 0.0f,
                      platform::errors::InvalidArgument(
                          "The value 'ratio' for Op(class_center_sample) must "
                          "be greater than 0.0f, "
                          "but the value given is %d.",
                          ratio));
    PADDLE_ENFORCE_LE(
        ratio, 1.0f,
        platform::errors::InvalidArgument(
            "The value 'ratio' for Op(class_center_sample) must be less or "
            "equal to 1.0f, but the value given is %d.",
            ratio));

    auto place = in->place();

    // remaped label
    out->Resize(in->dims());
    out->set_lod(in->lod());

    T* out_data = out->mutable_data<T>(context.GetPlace());
    framework::Tensor cpu_out_tensor;
    if (platform::is_gpu_place(place) || platform::is_xpu_place(place)) {
      cpu_out_tensor.Resize(out->dims());
      out_data = cpu_out_tensor.mutable_data<T>(platform::CPUPlace());
    }

    auto* in_data = in->data<T>();
    int64_t numel = in->numel();

    framework::Tensor cpu_in_tensor;
    if (platform::is_gpu_place(place) || platform::is_xpu_place(place)) {
      TensorCopySync(*in, platform::CPUPlace(), &cpu_in_tensor);
      in_data = cpu_in_tensor.data<T>();
    }

    // get unique class exclude ignore class
    std::set<T> unique_label;
    for (int64_t i = 0; i < numel; ++i) {
      if (in_data[i] == ignore_label) continue;
      unique_label.insert(in_data[i]);
    }

    std::uniform_int_distribution<T> dist(0, num_class - 1);
    auto engine = framework::GetCPURandomEngine(seed);
    size_t num_sample_class = static_cast<size_t>(num_class * ratio);

    // sample randomly
    while (unique_label.size() < num_sample_class) {
      unique_label.insert(dist(*engine));
    }

    // sampled class
    std::vector<int64_t> sampled_class_dim(1, unique_label.size());
    sampled_class->Resize(framework::make_ddim(sampled_class_dim));
    sampled_class->set_lod(in->lod());
    T* sampled_class_data = sampled_class->mutable_data<T>(context.GetPlace());
    framework::Tensor cpu_sampled_class_tensor;
    if (platform::is_gpu_place(place) || platform::is_xpu_place(place)) {
      cpu_sampled_class_tensor.Resize(sampled_class->dims());
      sampled_class_data =
          cpu_sampled_class_tensor.mutable_data<T>(platform::CPUPlace());
    }

    // constrcut a lookup table
    std::map<T, T> new_class_dict;
    T idx = 0;
    for (auto& t : unique_label) {
      new_class_dict[t] = idx;
      sampled_class_data[idx] = t;
      idx++;
    }

    // remap the input label to sampled class
    for (int64_t i = 0; i < numel; ++i) {
      if (in_data[i] == ignore_label) {
        out_data[i] = ignore_label;
      } else {
        out_data[i] = new_class_dict[in_data[i]];
      }
    }

    if (platform::is_gpu_place(place) || platform::is_xpu_place(place)) {
      TensorCopySync(cpu_out_tensor, place, out);
      TensorCopySync(cpu_sampled_class_tensor, place, sampled_class);
    }
  }
};

}  // namespace operators
}  // namespace paddle
