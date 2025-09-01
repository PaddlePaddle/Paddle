// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// #The file has been adapted from pytorch project
// #Licensed under  BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/common/exception.h"
#include "paddle/common/macros.h"

namespace c10 {
#define TORCH_CHECK(COND, ...) PD_CHECK(COND, ##__VA_ARGS__);
#define TORCH_INTERNAL_ASSERT(COND, ...) PD_CHECK(COND, ##__VA_ARGS__);
}  // namespace c10

enum class C10ErrorType {
  NotImplementedError,
  Error,
};

constexpr auto NotImplementedError = C10ErrorType::NotImplementedError;
constexpr auto Error = C10ErrorType::Error;

inline void C10ThrowImpl(C10ErrorType err_type, const std::string& msg) {
  switch (err_type) {
    case C10ErrorType::NotImplementedError:
      PADDLE_THROW(common::errors::Unimplemented(msg));
      break;
    case C10ErrorType::Error:
      PADDLE_THROW(common::errors::InvalidArgument(msg));
      break;
    default:
      PADDLE_THROW(common::errors::Fatal("Unknown error type: " + msg));
  }
}

#define C10_THROW_ERROR(err_type, msg) C10ThrowImpl(err_type, msg)
