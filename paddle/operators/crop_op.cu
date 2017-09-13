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
#include "paddle/operators/crop_op.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, int D>
__global__ void CropKernel(const int N, const int64_t* out_shape,
                           const int64_t* x_shape, const int* crop_rules,
                           const T* x_data, T* out_data) {
  CUDA_1D_KERNEL_LOOP(index, N) {
    // int64_t dim_size = out_shape.size();
    int64_t pos[D];

    for (int64_t i = D - 1; i >= 0; --i) {
      pos[i] = (index % out_shape[i]) + crop_rules[i * 2];
      index = index / out_shape[i];
    }

    int64_t result = pos[0];
    for (size_t i = 1; i < D; ++i) {
      result = result * x_shape[i] + pos[i];
    }

    out_data[index] = x_data[result];
  }
}

template <typename T, int D>
void CropCUDAFunctoin(const framework::ExecutionContext& context) {
  auto* x = context.Input<Tensor>("X");
  auto* out = context.Output<Tensor>("Out");
  auto x_data = x->data<T>();
  T* out_data = out->mutable_data<T>(paddle::platform::CPUPlace());
  auto x_dims = x->dims();
  auto out_dims = out->dims();
  int64_t out_count = framework::product(out_dims);
  int64_t* x_shape = &(framework::vectorize(x_dims))[0];
  int64_t* out_shape = &(framework::vectorize(out_dims))[0];

  auto offsets = context.op().Attr<std::vector<int>>("offsets");
  PADDLE_ENFORCE_EQ(
      x_dims.size(), offsets.size(),
      "Offsets size should be equal to dimension size of input tensor.");

  int crop_rules[D * 2];
  for (size_t i = 0; i < x_dims.size(); ++i) {
    crop_rules[i * 2] = offsets[i];
    crop_rules[i * 2 + 1] = x_dims[i] - out_dims[i] - offsets[i];
  }

  int n = out_dims[0];
  int d = out_dims[1];
  int block = 512;
  int grid = (n * d + block - 1) / block;

  CropKernel<T, D><<<grid, block>>>(out_count, out_shape, x_shape, crop_rules,
                                    x_data, out_data);
}

template <typename T>
class CropOpCUDAKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    size_t rank = context.Input<Tensor>("X")->dims().size();
    switch (rank) {
      case 1:
        CropCUDAFunctoin<T, 1>(context);
        break;
      case 2:
        CropCUDAFunctoin<T, 2>(context);
        break;
      case 3:
        CropCUDAFunctoin<T, 3>(context);
        break;
      case 4:
        CropCUDAFunctoin<T, 4>(context);
        break;
      case 5:
        CropCUDAFunctoin<T, 5>(context);
        break;
      case 6:
        CropCUDAFunctoin<T, 6>(context);
        break;
      default:
        PADDLE_THROW(
            "CropOp only support tensors with no more than 6 dimensions.");
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(crop, ops::CropOpCUDAKernel<float>);
REGISTER_OP_GPU_KERNEL(crop_grad,
                       ops::CropGradKernel<paddle::platform::GPUPlace, float>);
