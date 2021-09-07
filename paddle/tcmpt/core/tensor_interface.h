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

#include "paddle/tcmpt/core/backend.h"
#include "paddle/tcmpt/core/dtype.h"
#include "paddle/tcmpt/core/layout.h"

namespace paddle {
namespace framework {
class DDim;
}
namespace platform {
class Place;
}
}

namespace pt {

// TODO(chenweihang): Use the existing DDim directly?
// or design a abstract interface of DDim?
using DDim = paddle::framework::DDim;

// TODO(chenweihang): Use the existing Place directly?
// or design a abstract interface of Place?
using Place = paddle::platform::Place;

/**
 * The abstract class of Tensor implemention, it needs to define its basic
 * behavior through inherited classes.
 *
 * TensorInterface allows Tensor to uniformly access various different
 * TensorImpls within the framework. It will not be used as a kernel argument,
 * but only contains the interfaces supported by various TensorImpls.
 * In extreme cases, it can be an empty base class.
 *
 * If we don't use TensorInterface, we may need to use shared_ptr<void>
 * to unify Tensor's API.
 */
class TensorInterface {
 public:
  // Not allowed to initialize a tensor without descriptive metadata
  TensorInterface() = default;

  TensorInterface(const TensorInterface&) = delete;
  TensorInterface& operator=(const TensorInterface&) = delete;
  TensorInterface(TensorInterface&&) = delete;
  TensorInterface& operator=(TensorInterface&&) = delete;

  virtual ~TensorInterface() {}

  virtual int64_t numel() const = 0;

  virtual DDim dims() const = 0;

  virtual DataType type() const = 0;

  virtual DataLayout layout() const = 0;

  virtual Place place() const = 0;

  virtual Backend backend() const = 0;

  virtual bool initialized() const = 0;
};

}  // namespace pt
