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

#include <string>

#include "paddle/pten/hapi/include/backend_set.h"

namespace paddle {
namespace experimental {

struct TensorSignature final {
  std::string name{""};
  BackendSet backend_set{Backend::CPU};

  TensorSignature() = default;

  // open default methods if needed
  TensorSignature& operator=(const TensorSignature&) = delete;
  TensorSignature& operator=(TensorSignature&&) = delete;
  TensorSignature(const TensorSignature&) = delete;
  TensorSignature(TensorSignature&&) = delete;

  explicit TensorSignature(const std::string& t_name) : name(t_name) {}
  explicit TensorSignature(const Backend& t_backend) : backend_set(t_backend) {}
  explicit TensorSignature(const BackendSet& t_backend_set)
      : backend_set(t_backend_set) {}
  TensorSignature(const std::string& t_name, const BackendSet& t_backend_set)
      : name(t_name), backend_set(t_backend_set) {}
  TensorSignature(const std::string& t_name, const Backend& t_backend)
      : name(t_name), backend_set(t_backend) {}
};

}  // namespace experimental
}  // namespace paddle
