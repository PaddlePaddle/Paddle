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

#pragma once

#include <string>

#include "paddle_infer_declare.h"  // NOLINT

#ifdef PADDLE_WITH_ONNXRUNTIME
#include "onnxruntime_c_api.h"    // NOLINT
#include "onnxruntime_cxx_api.h"  // NOLINT
#endif

namespace paddle_infer {

/// \brief  Experimental.
/// Strings for text data.
using Strings = std::vector<std::string>;

typedef void (*CallbackFunc)(void*);

#if defined(PADDLE_WITH_TESTING) && defined(PADDLE_WITH_INFERENCE_API_TEST)
class InferApiTesterUtils;
#endif

namespace contrib {
class TensorUtils;
}

namespace experimental {
class InternalUtils;
};

/// \brief Paddle data type.
enum DataType {
  FLOAT32,
  INT64,
  INT32,
  UINT8,
  INT8,
  FLOAT16,
  // TODO(Superjomn) support more data types if needed.
};

enum class PlaceType { kUNK = -1, kCPU, kGPU, kXPU, kNPU, kIPU };

enum class DataLayout { kUNK = -1, kAny, kNHWC, kNCHW };

/// \brief Represents an n-dimensional array of values.
/// The Tensor is used to store the input or output of the network.
/// Zero copy means that the tensor supports direct copy of host or device data
/// to device,
/// eliminating additional CPU copy. Tensor is only used in the
/// AnalysisPredictor.
/// It is obtained through PaddlePredictor::GetinputTensor()
/// and PaddlePredictor::GetOutputTensor() interface.
class PD_INFER_DECL Tensor {
 public:
  /// \brief Reset the shape of the tensor.
  /// Generally it's only used for the input tensor.
  /// Reshape must be called before calling mutable_data() or copy_from_cpu()
  /// \param shape The shape to set.
  void Reshape(const std::vector<int>& shape);

  /// \brief Experimental interface.
  /// Reset the shape of the Strings tensor.
  /// Generally it's only used for the input tensor.
  /// Reshape must be called before calling
  /// ZeroCopyStringTensorCreate() or PaddleInferTensorCreate()
  /// \param shape The shape to set.
  void ReshapeStrings(const std::size_t& shape);

  /// \brief Get the memory pointer in CPU or GPU with specific data type.
  /// Please Reshape the tensor first before call this.
  /// It's usually used to get input data pointer.
  /// \param place The place of the tensor.
  template <typename T>
  T* mutable_data(PlaceType place);

  /// \brief Get the memory pointer directly.
  /// It's usually used to get the output data pointer.
  /// \param[out] place To get the device type of the tensor.
  /// \param[out] size To get the data size of the tensor.
  /// \return The tensor data buffer pointer.
  template <typename T>
  T* data(PlaceType* place, int* size) const;

  /// \brief Copy the host memory to tensor data.
  /// It's usually used to set the input tensor data.
  /// \param data The pointer of the data, from which the tensor will copy.
  template <typename T>
  void CopyFromCpu(const T* data);

  /// \brief Share the data with tensor data.
  /// It's usually used to set the tensor data.
  /// \param data The pointer of the data, from which the tensor will share.
  /// \param shape The shape of data.
  /// \param place The place of data.
  /// \param layout The layout of data. Only NCHW is supported now.
  template <typename T>
  void ShareExternalData(const T* data, const std::vector<int>& shape,
                         PlaceType place,
                         DataLayout layout = DataLayout::kNCHW);

  /// \brief Experimental interface.
  /// It's usually used to set the input tensor data with Strings data type.
  /// \param data The pointer of the data, from which the tensor will copy.
  void CopyStringsFromCpu(const paddle_infer::Strings* data);

  /// \brief Copy the tensor data to the host memory.
  /// It's usually used to get the output tensor data.
  /// \param[out] data The tensor will copy the data to the address.
  template <typename T>
  void CopyToCpu(T* data) const;

  /// \brief Copy the tensor data to the host memory asynchronously.
  /// \param[out] data The tensor will copy the data to the address.
  /// \param[out] exec_stream The tensor will excute copy in this stream(Only
  /// GPU CUDA stream suppported now).
  template <typename T>
  void CopyToCpuAsync(T* data, void* exec_stream) const;

  /// \brief Copy the tensor data to the host memory asynchronously.
  /// \param[out] data The tensor will copy the data to the address.
  /// \param[out] cb Callback function cb(cb_params) will be executed on the
  /// host after all currently enqueued items in the stream have completed .
  template <typename T>
  void CopyToCpuAsync(T* data, CallbackFunc cb, void* cb_params) const;

  /// \brief Return the shape of the Tensor.
  std::vector<int> shape() const;

  /// \brief Set lod info of the tensor.
  /// More about LOD can be seen here:
  ///  https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/basic_concept/lod_tensor.html#lodtensor
  /// \param x the lod info.
  void SetLoD(const std::vector<std::vector<size_t>>& x);
  /// \brief Return the lod info of the tensor.
  std::vector<std::vector<size_t>> lod() const;
  /// \brief Return the name of the tensor.
  const std::string& name() const;

  /// \brief Return the data type of the tensor.
  /// It's usually used to get the output tensor data type.
  /// \return The data type of the tensor.
  DataType type() const;

  /// \brief Return the place type of the tensor.
  /// \return The place type of the tensor.
  PlaceType place() const;

 protected:
  explicit Tensor(void* scope);

  template <typename T>
  void* FindTensor() const;

  void SetPlace(PlaceType place, int device = -1);
  void SetName(const std::string& name);

  template <typename T>
  void CopyToCpuImpl(T* data, void* stream = nullptr, CallbackFunc cb = nullptr,
                     void* cb_params = nullptr) const;

  std::string name_;
  // The corresponding tensor pointer inside Paddle workspace is cached for
  // performance.
  mutable void* tensor_{nullptr};
  DataType dtype_;
  bool input_or_output_;
  void* scope_{nullptr};
  PlaceType place_;
  int device_;

#ifdef PADDLE_WITH_ONNXRUNTIME
  bool is_ort_tensor_{false};
  std::vector<int64_t> shape_;
  std::weak_ptr<Ort::IoBinding> binding_;
  int idx_{-1};

  void SetOrtMark(bool is_ort_tensor);

  void SetOrtBinding(const std::shared_ptr<Ort::IoBinding> binding);

  template <typename T>
  void ORTCopyFromCpu(const T* data);

  template <typename T>
  void ORTCopyToCpu(T* data) const;
#endif

  friend class paddle_infer::contrib::TensorUtils;
  friend class paddle_infer::experimental::InternalUtils;
#if defined(PADDLE_WITH_TESTING) && defined(PADDLE_WITH_INFERENCE_API_TEST)
  friend class paddle_infer::InferApiTesterUtils;
#endif
};

}  // namespace paddle_infer
