/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include <paddle/fluid/framework/data_type.h>
#include <paddle/fluid/framework/operator.h>
#include <map>
#include <string>
#include <vector>
#include "paddle/fluid/framework/framework.pb.h"

namespace paddle {
namespace operators {

class MLUCnnlTensorDescPool {
 public:
  cnnlTensorDescriptor_t Pop() {
    cnnlTensorDescriptor_t raw_desc;
    if (q_.try_dequeue(raw_desc)) {
      return raw_desc;
    } else {
      cnnlCreateTensorDescriptor(&raw_desc);
      return raw_desc;
    }
  }

  void Recycle(cnnlTensorDescriptor_t desc) {
    cnnlResetTensorDescriptor(desc);
    q_.enqueue(desc);
  }

  ~MLUCnnlTensorDescPool() {
    auto size = q_.size_approx();
    if (size > 0) {
      std::vector<cnnlTensorDescriptor_t> vec(size);
      q_.try_dequeue_bulk(vec.data(), size);
      for (auto desc : vec) {
        cnnlDestroyTensorDescriptor(desc);
      }
    }
  }

 private:
  moodycamel::ConcurrentQueue<cnnlTensorDescriptor_t> q_;
};

static MLUCnnlTensorDescPool g_cnnl_tensor_desc_pool;

MLUCnnlTensorDesc &MLUCnnlTensorDesc::operator=(MLUCnnlTensorDesc &&rhs) {
  if (raw_tensor_desc) {
    g_cnnl_tensor_desc_pool.Recycle(raw_tensor_desc);
  }
  raw_tensor_desc = rhs.raw_tensor_desc;
  rhs.raw_tensor_desc = nullptr;
  return *this;
}

MLUCnnlTensorDesc::MLUCnnlTensorDesc(const int tensor_dim,
                                     const int dim_sizes[],
                                     const cnnlDataType_t tensor_dtype) {
  raw_tensor_desc = g_cnnl_tensor_desc_pool.Pop();
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetTensorDescriptor(
      raw_tensor_desc, CNNL_LAYOUT_ARRAY, tensor_dtype, tensor_dim, dim_sizes));
}

MLUCnnlTensorDesc::MLUCnnlTensorDesc(const int tensor_dim,
                                     const int dim_sizes[],
                                     const cnnlDataType_t tensor_dtype,
                                     const cnnlTensorLayout_t layout) {
  raw_tensor_desc = g_cnnl_tensor_desc_pool.Pop();
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetTensorDescriptor(
      raw_tensor_desc, layout, tensor_dtype, tensor_dim, dim_sizes));
}

MLUCnnlTensorDesc::MLUCnnlTensorDesc(const int tensor_dim,
                                     const int dim_sizes[],
                                     const cnnlDataType_t tensor_dtype,
                                     int position)
    : MLUCnnlTensorDesc(tensor_dim, dim_sizes, tensor_dtype) {
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptorPosition(raw_tensor_desc, position));
}

MLUCnnlTensorDesc::MLUCnnlTensorDesc(const int tensor_dim,
                                     const int64_t dim_sizes[],
                                     const cnnlDataType_t tensor_dtype) {
  std::vector<int> dim_sizes_int32(tensor_dim);
  std::vector<int64_t>::const_iterator int64_cbegin(dim_sizes);
  std::vector<int64_t>::const_iterator int64_cend(dim_sizes + tensor_dim);
  std::transform(int64_cbegin, int64_cend, dim_sizes_int32.begin(),
                 &CheckedNarrowing<int64_t, int>);
  raw_tensor_desc = g_cnnl_tensor_desc_pool.Pop();
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptor(raw_tensor_desc, CNNL_LAYOUT_ARRAY, tensor_dtype,
                              tensor_dim, dim_sizes_int32.data()));
}

MLUCnnlTensorDesc::MLUCnnlTensorDesc(const int tensor_dim,
                                     const int64_t dim_sizes[],
                                     const cnnlDataType_t tensor_dtype,
                                     const cnnlTensorLayout_t layout) {
  std::vector<int> dim_sizes_int32(tensor_dim);
  std::vector<int64_t>::const_iterator int64_cbegin(dim_sizes);
  std::vector<int64_t>::const_iterator int64_cend(dim_sizes + tensor_dim);
  std::transform(int64_cbegin, int64_cend, dim_sizes_int32.begin(),
                 &CheckedNarrowing<int64_t, int>);
  raw_tensor_desc = g_cnnl_tensor_desc_pool.Pop();
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetTensorDescriptor(raw_tensor_desc, layout,
                                                     tensor_dtype, tensor_dim,
                                                     dim_sizes_int32.data()));
}

MLUCnnlTensorDesc::MLUCnnlTensorDesc(const int tensor_dim,
                                     const int64_t dim_sizes[],
                                     const cnnlDataType_t tensor_dtype,
                                     int position) {
  std::vector<int> dim_sizes_int32(tensor_dim);
  std::vector<int64_t>::const_iterator int64_cbegin(dim_sizes);
  std::vector<int64_t>::const_iterator int64_cend(dim_sizes + tensor_dim);
  std::transform(int64_cbegin, int64_cend, dim_sizes_int32.begin(),
                 &CheckedNarrowing<int64_t, int>);
  raw_tensor_desc = g_cnnl_tensor_desc_pool.Pop();
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptor(raw_tensor_desc, CNNL_LAYOUT_ARRAY, tensor_dtype,
                              tensor_dim, dim_sizes_int32.data()));
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptorPosition(raw_tensor_desc, position));
}

MLUCnnlTensorDesc::MLUCnnlTensorDesc(const Tensor &tensor,
                                     const cnnlTensorLayout_t layout,
                                     const cnnlDataType_t tensor_dtype) {
  auto dims = framework::vectorize<int>(tensor.dims());
  int tensor_dim = dims.size();
  raw_tensor_desc = g_cnnl_tensor_desc_pool.Pop();
  if (tensor_dim == 0) {
    int scalar_dims[1] = {1};
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetTensorDescriptor(
        raw_tensor_desc, layout, tensor_dtype, 1, scalar_dims));
  } else {
    std::vector<int> tensor_dim_sizes_int(dims.begin(), dims.end());
    PADDLE_ENFORCE_MLU_SUCCESS(
        cnnlSetTensorDescriptor(raw_tensor_desc, layout, tensor_dtype,
                                tensor_dim, tensor_dim_sizes_int.data()));
  }
}

MLUCnnlTensorDesc::MLUCnnlTensorDesc(const Tensor &tensor,
                                     cnnlTensorLayout_t layout,
                                     const cnnlDataType_t tensor_dtype,
                                     int position)
    : MLUCnnlTensorDesc(tensor, layout, tensor_dtype) {
  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlSetTensorDescriptorPosition(raw_tensor_desc, position));
}

MLUCnnlTensorDesc::MLUCnnlTensorDesc(const Tensor &tensor,
                                     cnnlTensorLayout_t layout,
                                     const cnnlDataType_t tensor_dtype,
                                     int position, float scale)
    : MLUCnnlTensorDesc(tensor, layout, tensor_dtype) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetTensorDescriptorPositionAndScale(
      raw_tensor_desc, position, scale));
}

MLUCnnlTensorDesc::~MLUCnnlTensorDesc() {
  if (raw_tensor_desc) {
    g_cnnl_tensor_desc_pool.Recycle(raw_tensor_desc);
  }
}

MLUCnnlActivationDesc::MLUCnnlActivationDesc(
    const cnnlActivationMode_t act_mode, const float ceof) {
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlCreateActivationDescriptor(&active_desc_));
  PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetActivationDescriptor(
      active_desc_, act_mode, CNNL_NOT_PROPAGATE_NAN, ceof));
}

const cnnlActivationDescriptor_t MLUCnnlActivationDesc::get() const {
  return active_desc_;
}

MLUCnnlActivationDesc::~MLUCnnlActivationDesc() {
  if (active_desc_) {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlDestroyActivationDescriptor(active_desc_));
  }
}

/* static */ void MLUCnnl::Active(const platform::MLUDeviceContext &ctx,
                                  cnnlActivationDescriptor_t active_desc,
                                  const cnnlTensorDescriptor_t input_desc,
                                  const void *input,
                                  const cnnlTensorDescriptor_t output_desc,
                                  void *output) {
  cnnlHandle_t handle = ctx.cnnl_handle();

  PADDLE_ENFORCE_MLU_SUCCESS(cnnlActivationForward(
      handle, active_desc, NULL, input_desc, input, NULL, output_desc, output));
}

/* static */ void MLUCnnl::ActiveGrad(
    const platform::MLUDeviceContext &ctx,
    cnnlActivationDescriptor_t active_desc, const void *alpha, const void *beta,
    const cnnlTensorDescriptor_t y_desc, const void *y,
    const cnnlTensorDescriptor_t diff_y_desc, const void *diff_y,
    const cnnlTensorDescriptor_t x_desc, const void *x,
    const cnnlTensorDescriptor_t diff_x_desc, void *diff_x) {
  cnnlHandle_t handle = ctx.cnnl_handle();

  PADDLE_ENFORCE_MLU_SUCCESS(
      cnnlActivationBackward(handle, active_desc, alpha, y_desc, y, diff_y_desc,
                             diff_y, x_desc, x, beta, diff_x_desc, diff_x));
}

}  // namespace operators
}  // namespace paddle
