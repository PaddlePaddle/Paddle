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

#pragma once

#include "paddle/fluid/extension/include/device.h"
#include "paddle/fluid/extension/include/dtype.h"
#include <memory>

namespace paddle {

class CustomTensor{
public:
    /// \brief Construct a CustomTensor on None Place for CustomOp.
    /// Generally it's only used for user to create CustomTensor.
    explicit CustomTensor(PaddlePlace place);
    explicit CustomTensor(void* raw_tensor);
    /// \brief Reset the shape of the tensor.
    /// Generally it's only used for the input tensor.
    /// Reshape must be called before calling mutable_data() or copy_from_cpu()
    /// \param shape The shape to set.
    void Reshape(const std::vector<int>& shape);

    /// \brief Get the memory pointer in CPU or GPU with specific data type.
    /// Please Reshape the tensor first before call this.
    /// It's usually used to get input data pointer.
    /// \param place The place of the tensor this will override the original place
    /// of current tensor.
    template <typename T>
    T* mutable_data(const PaddlePlace& place);

    /// \brief Get the memory pointer in CPU or GPU with specific data type.
    /// Please Reshape the tensor first before call this.
    /// It's usually used to get input data pointer.
    template <typename T>
    T* mutable_data();

    /// \brief Get the memory pointer directly.
    /// It's usually used to get the output data pointer.
    /// \return The tensor data buffer pointer.
    template <typename T>
    T* data() const;

    /// \brief Copy the host memory to tensor data.
    /// It's usually used to set the input tensor data.
    /// \param data The pointer of the data, from which the tensor will copy.
    template <typename T>
    void copy_from_cpu(const T* data);

    /// \brief Copy the tensor data to the host memory.
    /// It's usually used to get the output tensor data.
    /// \param[out] data The tensor will copy the data to the address.
    template <typename T>
    void copy_to_cpu(T* data);

    /// \brief Return the shape of the Tensor.
    std::vector<int> shape() const;

    /// \brief Set lod info of the tensor.
    /// More about LOD can be seen here:
    ///  https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/basic_concept/lod_tensor.html#lodtensor
    /// \param x the lod info.
    void SetLoD(const std::vector<std::vector<size_t>>& x);
    /// \brief Return the lod info of the tensor.
    std::vector<std::vector<size_t>> lod() const;

    /// \brief Return the data type of the tensor.
    /// It's usually used to get the output tensor data type.
    /// \return The data type of the tensor.
    PaddleDType type() const;


    /// \brief Share data TO another tensor.
    /// Use this to pass tensor from op to op
    /// \return void.
    void ShareDataTo(void* other);

    /// \brief Share data FROM another tensor.
    /// Use this to pass tensor from op to op
    /// \return void.
    void ShareDataFrom(void* other);

    /// \brief Get the size of current tensor.
    /// Use this method to get the size of tensor
    /// \return int64_t.
    int64_t size() const;

    /// \brief Get the place of current tensor.
    /// Use this method to get the place of tensor
    /// \return Place.
    const PaddlePlace& place() const;

private:
    mutable std::shared_ptr<void> tensor_;
    mutable PaddlePlace place_;
};

}  // namespace paddle
