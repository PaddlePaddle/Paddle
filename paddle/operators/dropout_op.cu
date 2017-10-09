/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#define EIGEN_USE_GPU
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include "paddle/operators/dropout_op.h"

namespace paddle {
namespace operators {

template <typename T, typename AttrType>
struct MaskGenerator {
  AttrType dropout_prob;
  int seed;

  __host__ __device__ MaskGenerator(AttrType dropout_prob, int seed)
      : dropout_prob(dropout_prob), seed(seed) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed);
    thrust::uniform_real_distribution<AttrType> dist(0, 1);
    rng.discard(n);
    if (dist(rng) < dropout_prob) {
      return static_cast<T>(0);
    } else {
      return static_cast<T>(1);
    }
  }
};

// It seems that Eigen::Tensor::setRandom in GPU will SEGFAULT.
// Use std::random and thrust::random(thrust is a std library in CUDA) to
// implement uniform random.
template <typename Place, typename T, typename AttrType>
class GPUDropoutKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Output<Tensor>("Out");
    y->mutable_data<T>(context.GetPlace());
    AttrType dropout_prob = context.Attr<AttrType>("dropout_prob");

    auto X = EigenMatrix<T>::Reshape(*x, 1);
    auto Y = EigenMatrix<T>::Reshape(*y, 1);

    auto place = context.GetEigenDevice<Place>();
    if (context.Attr<bool>("is_training")) {
      auto* mask = context.Output<Tensor>("Mask");
      auto* mask_data = mask->mutable_data<T>(context.GetPlace());
      int size = framework::product(mask->dims());
      int seed = context.Attr<int>("seed");
      thrust::counting_iterator<unsigned int> index_sequence_begin(0);
      thrust::transform(index_sequence_begin, index_sequence_begin + size,
                        thrust::device_ptr<T>(mask_data),
                        MaskGenerator<T, AttrType>(dropout_prob, seed));
      auto M = EigenMatrix<T>::Reshape(*mask, 1);
      Y.device(place) = X * M;
    } else {
      Y.device(place) = X * dropout_prob;
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(
    dropout, ops::GPUDropoutKernel<paddle::platform::GPUPlace, float, float>);
REGISTER_OP_GPU_KERNEL(
    dropout_grad, ops::DropoutGradKernel<paddle::platform::GPUPlace, float>);
