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

#include "paddle/tcmpt/core/tensor_interface.h"

namespace pt {

/**
 * SpatialTensor represents a Tensor whose memory layout is different from
 * the typical Allocation (size+ptr).
 *
 * It needs to pass in a specific Allocation implementation when it is
 * instantiated.
 */

template <typename AllocationType>
class SpatialTensor : public TensorInterface {
 public:
  SpatialTensor(std::shared_ptr<AllocationType> allocation,
                std::unique_ptr<TensorMeta> meta,
                std::unique_ptr<TensorStatus> status)
      : allocation_(std::move(allocation)),
        meta_(std::move(meta)),
        status_(std::move(status)) {}

 private:
  std::shared_ptr<AllocationType> allocation_;
  std::unique_ptr<TensorMeta> meta_;
  std::unique_ptr<TensorStatus> status_;
};

template <typename AllocationType>
class MetalTensor : public SpatialTensor<AllocationType> {};

template <typename AllocationType>
class OpenCLTensor : public SpatialTensor<AllocationType> {};

}  // namespace pt
