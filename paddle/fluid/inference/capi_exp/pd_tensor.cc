// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/capi_exp/pd_tensor.h"

#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/capi_exp/pd_types.h"
#include "paddle/fluid/inference/capi_exp/pd_utils.h"
#include "paddle/fluid/inference/capi_exp/types_internal.h"
#include "paddle/fluid/inference/capi_exp/utils_internal.h"
#include "paddle/fluid/platform/enforce.h"

#define CHECK_AND_CONVERT_PD_TENSOR                              \
  PADDLE_ENFORCE_NOT_NULL(                                       \
      pd_tensor,                                                 \
      common::errors::InvalidArgument(                           \
          "The pointer of paddle tensor shouldn't be nullptr")); \
  auto& tensor = pd_tensor->tensor

extern "C" {

void PD_TensorDestroy(__pd_take PD_Tensor* pd_tensor) { delete pd_tensor; }
void PD_TensorReshape(__pd_keep PD_Tensor* pd_tensor,
                      size_t shape_size,
                      int32_t* shape) {
  CHECK_AND_CONVERT_PD_TENSOR;
  std::vector<int> shapes(shape_size);
  for (size_t index = 0; index < shape_size; ++index) {
    shapes[index] = shape[index];
  }
  tensor->Reshape(shapes);
}

#define REPEAT_ALL_DATA_TYPE(func)                             \
  func(float, Float) func(int64_t, Int64) func(int32_t, Int32) \
      func(uint8_t, Uint8) func(int8_t, Int8)

#define PD_TENSOR_MUTABLE_DATA_IMPL(type, Type)                                \
  type* PD_TensorMutableData##Type(__pd_keep PD_Tensor* pd_tensor,             \
                                   PD_PlaceType place) {                       \
    CHECK_AND_CONVERT_PD_TENSOR;                                               \
    return tensor->mutable_data<type>(paddle_infer::CvtToCxxPlaceType(place)); \
  }
REPEAT_ALL_DATA_TYPE(PD_TENSOR_MUTABLE_DATA_IMPL)
#undef PD_TENSOR_MUTABLE_DATA_IMPL

#define PD_TENSOR_DATA_IMPL(type, Type)                                        \
  type* PD_TensorData##Type(                                                   \
      __pd_keep PD_Tensor* pd_tensor, PD_PlaceType* place, int32_t* size) {    \
    CHECK_AND_CONVERT_PD_TENSOR;                                               \
    PADDLE_ENFORCE_NOT_NULL(place,                                             \
                            common::errors::InvalidArgument(                   \
                                "The pointer of place shouldn't be nullptr")); \
    PADDLE_ENFORCE_NOT_NULL(size,                                              \
                            common::errors::InvalidArgument(                   \
                                "The pointer of size shouldn't be nullptr"));  \
    paddle_infer::PlaceType cxx_place_type;                                    \
    int cxx_size;                                                              \
    type* data = tensor->data<type>(&cxx_place_type, &cxx_size);               \
    *place = paddle_infer::CvtFromCxxPlaceType(cxx_place_type);                \
    *size = static_cast<int32_t>(cxx_size);                                    \
    return data;                                                               \
  }
REPEAT_ALL_DATA_TYPE(PD_TENSOR_DATA_IMPL)
#undef PD_TENSOR_DATA_IMPL

#define PD_TENSOR_COPY_FROM_CPU_IMPL(type, Type)                  \
  void PD_TensorCopyFromCpu##Type(__pd_keep PD_Tensor* pd_tensor, \
                                  const type* data) {             \
    CHECK_AND_CONVERT_PD_TENSOR;                                  \
    tensor->CopyFromCpu<type>(data);                              \
  }
REPEAT_ALL_DATA_TYPE(PD_TENSOR_COPY_FROM_CPU_IMPL)
#undef PD_TENSOR_COPY_FROM_CPU_IMPL

#define PD_TENSOR_COPY_TO_CPU_IMPL(type, Type)                                \
  void PD_TensorCopyToCpu##Type(__pd_keep PD_Tensor* pd_tensor, type* data) { \
    CHECK_AND_CONVERT_PD_TENSOR;                                              \
    tensor->CopyToCpu<type>(data);                                            \
  }
REPEAT_ALL_DATA_TYPE(PD_TENSOR_COPY_TO_CPU_IMPL)
#undef PD_TENSOR_COPY_TO_CPU_IMPL

#undef REPEAT_ALL_DATA_TYPE

__pd_give PD_OneDimArrayInt32* PD_TensorGetShape(
    __pd_keep PD_Tensor* pd_tensor) {
  CHECK_AND_CONVERT_PD_TENSOR;
  return paddle_infer::CvtVecToOneDimArrayInt32(tensor->shape());
}
void PD_TensorSetLod(__pd_keep PD_Tensor* pd_tensor,
                     __pd_keep PD_TwoDimArraySize* lod) {
  CHECK_AND_CONVERT_PD_TENSOR;
  tensor->SetLoD(paddle_infer::CvtTwoDimArrayToVecSize(lod));
}
__pd_give PD_TwoDimArraySize* PD_TensorGetLod(__pd_keep PD_Tensor* pd_tensor) {
  CHECK_AND_CONVERT_PD_TENSOR;
  return paddle_infer::CvtVecToTwoDimArraySize(tensor->lod());
}
const char* PD_TensorGetName(__pd_keep PD_Tensor* pd_tensor) {
  CHECK_AND_CONVERT_PD_TENSOR;
  return tensor->name().c_str();
}
PD_DataType PD_TensorGetDataType(__pd_keep PD_Tensor* pd_tensor) {
  CHECK_AND_CONVERT_PD_TENSOR;
  return paddle_infer::CvtFromCxxDatatype(tensor->type());
}

}  // extern "C"
