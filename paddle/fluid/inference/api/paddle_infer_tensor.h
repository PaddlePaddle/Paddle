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
  // TODO(Superjomn) support more data types if needed.
};

enum class PlaceType { kUNK = -1, kCPU, kGPU, kXPU };

/// \brief Memory manager for HostTensor.
///
/// The HostBuffer holds a buffer for data input or output. The memory can be
/// allocated by user or by HostBuffer itself, but in any case, the HostBuffer
/// should be reused for better performance.
///
/// For user allocated memory, the following API can be used:
/// - HostBuffer(void* data, size_t length) to set an external memory by
/// specifying the memory address and length.
/// - Reset(void* data, size_t length) to reset the HostBuffer with an external
/// memory.
/// ATTENTION, for user allocated memory, deallocation should be done by users
/// externally after the program finished. The HostBuffer won't do any
/// allocation
/// or deallocation.
///
/// To have the HostBuffer allocate and manage the memory:
/// - HostBuffer(size_t length) will allocate a memory of size `length`.
/// - Resize(size_t length) resize the memory to no less than `length`,
/// ATTENTION
///  if the allocated memory is larger than `length`, nothing will done.
///
/// Usage:
///
/// Let HostBuffer manage the memory internally.
/// \code{cpp}
/// const int num_elements = 128;
/// HostBuffer buf(num_elements/// sizeof(float));
/// \endcode
///
/// Or
/// \code{cpp}
/// HostBuffer buf;
/// buf.Resize(num_elements/// sizeof(float));
/// \endcode
/// Works the exactly the same.
///
/// One can also make the `HostBuffer` use the external memory.
/// \code{cpp}
/// HostBuffer buf;
/// void* external_memory = new float[num_elements];
/// buf.Reset(external_memory, num_elements*sizeof(float));
/// ...
/// delete[] external_memory; // manage the memory lifetime outside.
/// \endcode
///
class PD_INFER_DECL HostBuffer {
 public:
  ///
  /// \brief HostBuffer allocate memory internally, and manage it.
  ///
  /// \param[in] length The length of data.
  ///
  explicit HostBuffer(size_t length)
      : data_(new char[length]), length_(length), memory_owned_(true) {}
  ///
  /// \brief Set external memory, the HostBuffer won't manage it.
  ///
  /// \param[in] data The start address of the external memory.
  /// \param[in] length The length of data.
  ///
  HostBuffer(void* data, size_t length)
      : data_(data), length_(length), memory_owned_{false} {}
  ///
  /// \brief Copy only available when memory is managed externally.
  ///
  /// \param[in] other another `HostBuffer`
  ///
  explicit HostBuffer(const HostBuffer& other);
  ///
  /// \brief Resize the memory.
  ///
  /// \param[in] length The length of data.
  ///
  void Resize(size_t length);
  ///
  /// \brief Reset to external memory, with address and length set.
  ///
  /// \param[in] data The start address of the external memory.
  /// \param[in] length The length of data.
  ///
  void Reset(void* data, size_t length);
  ///
  /// \brief Tell whether the buffer is empty.
  ///
  bool empty() const { return length_ == 0; }
  ///
  /// \brief Get the data's memory address.
  ///
  void* data() const { return data_; }
  ///
  /// \brief Get the memory length.
  ///
  size_t length() const { return length_; }

  ~HostBuffer() { Free(); }
  HostBuffer& operator=(const HostBuffer&);
  HostBuffer& operator=(HostBuffer&&);
  HostBuffer() = default;
  HostBuffer(HostBuffer&& other);

 private:
  void Free();
  void* data_{nullptr};  ///< pointer to the data memory.
  size_t length_{0};     ///< number of memory bytes.
  bool memory_owned_{true};
};

///
/// \brief Basic input and output data structure for PaddlePredictor.
///
struct PD_INFER_DECL HostTensor {
  HostTensor() = default;
  std::string name;  ///<  variable name.
  std::vector<int> shape;
  HostBuffer data;  ///<  blob of data.
  DataType dtype;
  std::vector<std::vector<size_t>> lod;  ///<  Tensor+LoD equals LoDTensor
};

/// \brief Represents an n-dimensional array of values.
/// TensorHandle is used to refer to a tensor in the predictor.
/// It is a handle, and the user cannot construct it directly.
/// It supports direct copy of host or device data to device,
/// eliminating additional CPU copy. It is obtained through Predictor::
/// GetInputHandle() and PaddlePredictor::GetOutputHandle() interface.
class PD_INFER_DECL TensorHandle {
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

  /// \brief Copy the host memory to the referenced tensor.
  /// It's usually used to set the input tensor data.
  /// \param data The pointer of the data, from which the tensor will copy.
  template <typename T>
  void CopyFromCpu(const T* data);

  /// \brief Copy the referenced tensor data to the host memory.
  /// It's usually used to get the output tensor data.
  /// \param[out] data The tensor will copy the data to the address.
  template <typename T>
  void CopyToCpu(T* data);

  /// \brief Return the shape of the tensor.
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
  explicit TensorHandle(void* scope);
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
