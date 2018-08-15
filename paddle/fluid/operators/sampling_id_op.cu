/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */
#include <thrust/random.h>
#include <thrust/transform.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

template <typename T>
struct UniformGenerator {
  T min_, max_;
  unsigned int seed_;

  __host__ __device__ UniformGenerator(T min, T max, int seed)
      : min_(min), max_(max), seed_(seed) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::uniform_real_distribution<T> dist(min_, max_);
    rng.discard(n);
    return dist(rng);
  }
};

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class SamplingIdGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("X");
    const int batch_size = static_cast<int>(input->dims()[0]);
    const int width = static_cast<int>(input->dims()[1]);

    std::vector<T> ins_vector;
    framework::TensorToVector(*input, context.device_context(), &ins_vector);

    unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));
    if (seed == 0) {
      std::random_device rd;
      seed = rd();
    }
    T min = static_cast<T>(context.Attr<float>("min"));
    T max = static_cast<T>(context.Attr<float>("max"));

    std::vector<T> ids(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      T r = UniformGenerator<T>(min, max, seed);
      int idx = width - 1;
      for (int j = 0; j < width; ++j) {
        if ((r -= ins_vector[i * width + j]) < 0) {
          idx = j;
          break;
        }
      }
      ids[i] = ins_vector[i * width + idx];
    }

    std::vector<int64_t> out_dim;
    out_dim.push_back(static_cast<int64_t>(batch_size));

    Tensor* output = context.Output<Tensor>("Out");
    output->Resize(framework::make_ddim(out_dim));
    output->mutable_data<T>(context.GetPlace());
    framework::TensorFromVector(ids, context.device_context(), output);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(sampling_id,
                        paddle::operators::SamplingIdGPUKernel<float>,
                        paddle::operators::SamplingIdGPUKernel<double>);
