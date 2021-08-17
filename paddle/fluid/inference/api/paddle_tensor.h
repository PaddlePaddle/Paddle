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

#include "paddle_infer_declare.h"  // NOLINT

namespace paddle_infer {

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

enum class PlaceType { kUNK = -1, kCPU, kGPU, kXPU, kNPU };

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

  /// \brief Copy the tensor data to the host memory.
  /// It's usually used to get the output tensor data.
  /// \param[out] data The tensor will copy the data to the address.
  template <typename T>
  void CopyToCpu(T* data);

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

 protected:
  explicit Tensor(void* scope);
  void* FindTensor() const;
  void SetPlace(PlaceType place, int device = -1);
  void SetName(const std::string& name);

  std::string name_;
  // The corresponding tensor pointer inside Paddle workspace is cached for
  // performance.
  mutable void* tensor_{nullptr};
  DataType dtype_;
  bool input_or_output_;
  void* scope_{nullptr};
  PlaceType place_;
  int device_;
};

}  // namespace paddle_infer
