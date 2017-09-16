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

#pragma once
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <random>
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class CPUDropoutKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Output<Tensor>("Out");
    auto* mask = context.Output<Tensor>("Mask");
    T* mask_data = mask->mutable_data<T>(context.GetPlace());
    T* y_data = y->mutable_data<T>(context.GetPlace());
    const T* x_data = x->data<T>();

    float dropout_prob = context.Attr<float>("dropout_prob");
    int seed = context.Attr<int>("seed");

    std::minstd_rand engine;
    engine.seed(seed);
    std::uniform_real_distribution<T> dist(0, 1);
    size_t size = framework::product(mask->dims());
    for (size_t i = 0; i < size; ++i) {
      if (dist(engine) < dropout_prob) {
        mask_data[i] = 0;
        y_data[i] = 0;
      } else {
        mask_data[i] = 1;
        y_data[i] = x_data[i];
      }
    }
    // TODO: add test phase logits.
  }
};

template <typename T>
struct MaskGenerator {
  float dropout_prob;
  int seed;

  __host__ __device__ MaskGenerator(float dropout_prob, int seed)
      : dropout_prob(dropout_prob), seed(seed) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed);
    thrust::uniform_real_distribution<T> dist(0, 1);
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
template <typename Place, typename T>
class GPUDropoutKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Output<Tensor>("Out");
    auto* mask = context.Output<Tensor>("Mask");
    y->mutable_data<T>(context.GetPlace());

    float dropout_prob = context.Attr<float>("dropout_prob");
    int seed = context.Attr<int>("seed");
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    int size = framework::product(mask->dims());
    T* mask_data = mask->mutable_data<T>(context.GetPlace());
    thrust::transform(index_sequence_begin, index_sequence_begin + size,
                      thrust::device_ptr<T>(mask_data),
                      MaskGenerator<T>(dropout_prob, seed));

    auto dims = x->dims();
    auto new_dims = framework::make_ddim({dims[0], size / dims[0]});
    auto X = EigenMatrix<T>::From(*x, new_dims);
    auto Y = EigenMatrix<T>::From(*y, new_dims);
    auto M = EigenMatrix<T>::From(*mask, new_dims);

    auto place = context.GetEigenDevice<Place>();
    Y.device(place) = X * M;
    // TODO: add test time logits.
  }
};

template <typename Place, typename T>
class DropoutGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* grad_y = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* mask = context.Input<Tensor>("Mask");
    grad_x->mutable_data<T>(context.GetPlace());

    auto dims = grad_x->dims();
    int size = static_cast<int>(framework::product(dims));
    auto new_dims = framework::make_ddim({dims[0], size / dims[0]});
    auto M = EigenMatrix<T>::From(*mask, new_dims);
    auto dX = EigenMatrix<T>::From(*grad_x, new_dims);
    auto dY = EigenMatrix<T>::From(*grad_y, new_dims);

    auto place = context.GetEigenDevice<Place>();
    dX.device(place) = dY * M;
    // TODO: add test time logits.
  }
};

}  // namespace operators
}  // namespace paddle
