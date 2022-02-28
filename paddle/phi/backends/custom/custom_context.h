/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <memory>
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/device_context.h"

namespace phi {

class CustomContext : public DeviceContext {
 public:
  explicit CustomContext(const CustomPlace&);

  virtual ~CustomContext();

  const Place& GetPlace() const override;

  /*! \brief  Return stream in the device context. */
  void* stream() const;

  // Wait for all operations completion in the stream.
  void Wait() const override;

 public:
  // NOTE: DeviceContext hold resources. Used in training scenarios.
  // The interface used by the training scene, DeviceContext will initialize
  // all resources and delete them when destructing.
  void Init();

 private:
  CustomContext();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace phi
