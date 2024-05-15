/* Copyright (c) 2023 Enflame. All Rights Reserved.

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

#include <cstdint>
#include <memory>
#include <string>

#include "paddle/fluid/platform/device/gcu/runtime/rt_utils.h"

namespace paddle {
namespace platform {
namespace gcu {
namespace runtime {

enum class ChipType {
  LEO,
  PAVO,
  PAVO_1C,
  DORADO,
  DORADO_2C,
  DORADO_3PG,
  LIBRA,
  SCORPIO,
  UNKNOW,
};

struct Stream;
struct Context {
  int device;

  std::shared_ptr<Stream> default_exe_stream;

  std::shared_ptr<Stream> default_dma_stream;

  static std::shared_ptr<Context> CreateContext(int device);

  static int VisibleDeviceCount();

  static std::string GlobalTargetName();

  static ChipType GlobalChipType();

  void Synchronize();

  std::string GetName() const;

  ChipType GetChipType() const;

  std::string GetTargetName() const;

  void Init(int device);

  ~Context();

  explicit Context(int device);

  Context() = delete;

  RT_DISALLOW_COPY_AND_ASSIGN(Context);
};

}  // namespace runtime
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
