// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <type_traits>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"

// TVM redefines some macros in GLOG, causing compilation error
// Define DMLC_GLOG_DEFINED to disable macros redefinition
#ifndef DMLC_GLOG_DEFINED
#define DMLC_GLOG_DEFINED
#endif

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

namespace paddle {
namespace framework {
namespace tvm {
namespace runtime {

class DLPackTensor {
 public:
  using LaneType = decltype(::DLTensor::dtype.lanes);  // uint16_t
  using ShapeType =
      std::remove_reference<decltype(::DLTensor::shape[0])>::type;  // int64_t

  // lanes is only used in CPU to enable vectorization
  explicit DLPackTensor(const Tensor& tensor, LaneType lanes = 1);

  /* TVM exposes PackedFunc class for runtime applications.
   * User can call PackedFunc like that:
   *
   *        // Load a pre-compiled *.so
   *        Module module = Module::LoadFromFile(library_name);
   *
   *        // Retrieve function inside *.so
   *        PackedFunc f = module.GetFunction(func_name);
   *
   *        // Invoke the functions
   *        f(x1, x2, ..., xn);
   *
   * PackedFunc is a data structure like std::function<void(Args...)>
   * which accepts variadic parameters. Parameters of PackedFunc could be any
   * of:
   *  1. Basic C++ type, such as bool, int, float, double, std::string etc...
   *  2. DLTensor *.
   *
   * However, PackedFunc does not distinguish what are inputs and what are
   * outputs inside its parameters, and cannot accept "const DLTensor*" as
   * parameter each though this parameter is exactly a const one. Therefore,
   * we can only return mutable DLTensor* here for implicit conversion.
   */
  operator ::DLTensor*() const { return &t_; }

 private:
  // DLTensor is a POD object with data buffer, dtype, shape...
  mutable ::DLTensor t_;

  // The shape in DLTensor is defined as int64_t*
  // Add this member to make TVMTensor init without heap allocation
  ShapeType shape_[9];
};

#ifdef PADDLE_WITH_CUDA
void SetStream(int device_id, const cudaStream_t& stream);
void SetStream(const platform::CUDAPlace& place);
#endif

template <typename Func, typename... Args>
inline auto RunFuncInDevice(const platform::Place& place, Func&& func,
                            Args&&... args)
    -> decltype(func(std::forward<Args>(args)...)) {
#ifdef PADDLE_WITH_CUDA
  if (platform::is_gpu_place(place)) {
    SetStream(boost::get<platform::CUDAPlace>(place));
  }
#endif
  return func(std::forward<Args>(args)...);
}

using namespace ::tvm::runtime;  // NOLINT

PackedFunc GetFuncFromLib(const std::string& lib_name,
                          const std::string& func_name);

}  // namespace runtime
}  // namespace tvm
}  // namespace framework
}  // namespace paddle
