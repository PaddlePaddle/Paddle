/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <cmath>
#include <memory>
#include <vector>

<<<<<<< HEAD
=======
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/eigen.h"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"
<<<<<<< HEAD
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
=======
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/utils/data_type.h"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

namespace phi {
namespace funcs {

<<<<<<< HEAD
template <typename T>
void BatchTranspose(T* output, const T* input, int batch, int m, int n);

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
template <typename DeviceContext, typename T>
struct TransposeNormal {
  // for dims >= 7 situation
  void operator()(const DeviceContext& context,
<<<<<<< HEAD
                  const phi::DenseTensor& in,
                  phi::DenseTensor* out,
=======
                  const paddle::framework::Tensor& in,
                  paddle::framework::Tensor* out,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                  const std::vector<int>& axis);
};

template <typename DeviceContext, typename T, int Rank>
struct Transpose {
  void operator()(const DeviceContext& context,
<<<<<<< HEAD
                  const phi::DenseTensor& in,
                  phi::DenseTensor* out,
=======
                  const paddle::framework::Tensor& in,
                  paddle::framework::Tensor* out,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                  const std::vector<int>& axis);
};

template <typename DeviceContext, typename T>
struct SetConstant {
  void operator()(const DeviceContext& context,
<<<<<<< HEAD
                  phi::DenseTensor* tensor,
=======
                  paddle::framework::Tensor* tensor,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                  T num);
};

#ifdef PADDLE_WITH_XPU
template <typename T>
struct SetConstant<XPUContext, T> {
<<<<<<< HEAD
  void operator()(const XPUContext& context, phi::DenseTensor* tensor, T num);
=======
  void operator()(const XPUContext& context,
                  paddle::framework::Tensor* tensor,
                  T num);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
};

template <typename T>
struct SetConstant<paddle::platform::XPUDeviceContext, T> {
  void operator()(const paddle::platform::XPUDeviceContext& context,
<<<<<<< HEAD
                  phi::DenseTensor* tensor,
=======
                  paddle::framework::Tensor* tensor,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                  T num);
};
#endif

template <typename Place>
void set_constant_with_place(const paddle::platform::DeviceContext& context,
<<<<<<< HEAD
                             phi::DenseTensor* tensor,
                             float value);

void set_constant(const paddle::platform::DeviceContext& context,
                  phi::DenseTensor* tensor,
=======
                             paddle::framework::Tensor* tensor,
                             float value);

void set_constant(const paddle::platform::DeviceContext& context,
                  paddle::framework::Tensor* tensor,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                  float value);

template <typename DeviceContext, typename T>
struct RowwiseAdd {
  void operator()(const DeviceContext& context,
<<<<<<< HEAD
                  const phi::DenseTensor& input,
                  const phi::DenseTensor& vec,
                  phi::DenseTensor* output);
=======
                  const paddle::framework::Tensor& input,
                  const paddle::framework::Tensor& vec,
                  paddle::framework::Tensor* output);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
};

template <typename DeviceContext, typename T>
struct ColwiseSum {
  void operator()(const DeviceContext& context,
<<<<<<< HEAD
                  const phi::DenseTensor& input,
                  phi::DenseTensor* vec);
=======
                  const paddle::framework::Tensor& input,
                  paddle::framework::Tensor* vec);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
};

template <typename DeviceContext, typename T>
struct RowwiseSum {
  void operator()(const DeviceContext& context,
<<<<<<< HEAD
                  const phi::DenseTensor& input,
                  phi::DenseTensor* vec);
=======
                  const paddle::framework::Tensor& input,
                  paddle::framework::Tensor* vec);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
};

template <typename DeviceContext, typename T>
struct RowwiseMean {
  void operator()(const DeviceContext& context,
<<<<<<< HEAD
                  const phi::DenseTensor& input,
                  phi::DenseTensor* vec);
=======
                  const paddle::framework::Tensor& input,
                  paddle::framework::Tensor* vec);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
};

#ifdef PADDLE_WITH_XPU
template <typename U>
struct TensorSetConstantXPU {
<<<<<<< HEAD
  TensorSetConstantXPU(phi::DenseTensor* tensor,
=======
  TensorSetConstantXPU(paddle::framework::Tensor* tensor,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                       U value,
                       paddle::platform::Place place)
      : tensor_(tensor), value_(value), place_(place) {}
  template <typename T>
  void apply() const {
    auto* begin = tensor_->mutable_data<T>(place_);
    int numel = tensor_->numel();
    std::unique_ptr<T[]> data_cpu(new T[numel]);
    std::fill(data_cpu.get(), data_cpu.get() + numel, static_cast<T>(value_));
    paddle::memory::Copy(place_,
                         begin,
                         phi::CPUPlace(),
                         static_cast<void*>(data_cpu.get()),
                         numel * sizeof(T));
  }
<<<<<<< HEAD
  phi::DenseTensor* tensor_;
=======
  paddle::framework::Tensor* tensor_;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  U value_;
  paddle::platform::Place place_;
};
#endif

template <typename Context, typename T>
inline void TransCompute(const int dim,
                         const Context& dev_ctx,
                         const DenseTensor& in,
                         DenseTensor* out,
                         const std::vector<int>& axis) {
  switch (dim) {
    case 1:
      Transpose<Context, T, 1> trans1;
      trans1(dev_ctx, in, out, axis);
      break;
    case 2:
      Transpose<Context, T, 2> trans2;
      trans2(dev_ctx, in, out, axis);
      break;
    case 3:
      Transpose<Context, T, 3> trans3;
      trans3(dev_ctx, in, out, axis);
      break;
    case 4:
      Transpose<Context, T, 4> trans4;
      trans4(dev_ctx, in, out, axis);
      break;
    case 5:
      Transpose<Context, T, 5> trans5;
      trans5(dev_ctx, in, out, axis);
      break;
    case 6:
      Transpose<Context, T, 6> trans6;
      trans6(dev_ctx, in, out, axis);
      break;
    default:
      // for dim >= 7 situation
      TransposeNormal<Context, T> trans_normal;
      trans_normal(dev_ctx, in, out, axis);
  }
}

}  // namespace funcs
}  // namespace phi
