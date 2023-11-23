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

#include "paddle/fluid/platform/device/gcu/runtime/rt_executable.h"
#include <memory>
#include <tuple>

#include "dtu/hlir/dispatch.h"
#include "dtu/hlir/types.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_resources.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace gcu {
namespace runtime {

Executable::Executable(topsExecutable_t executable) : tops_exec(executable) {
  PADDLE_ENFORCE_NOT_NULL(tops_exec,
                          paddle::platform::errors::InvalidArgument(
                              "Expect executable is not null."));
  ResourceMgr::GetInstance()->RTCounter("Executable", 1);
}

Executable::Executable(hlir::HlirDispatch *in_dispatch,
                       const std::vector<uint64_t> in_input_sizes,
                       const std::vector<uint64_t> in_output_sizes)
    : dispatch(in_dispatch),
      input_sizes(in_input_sizes),
      output_sizes(in_output_sizes) {
  PADDLE_ENFORCE_NOT_NULL(dispatch,
                          paddle::platform::errors::InvalidArgument(
                              "Expect dispatch is not null."));
  ResourceMgr::GetInstance()->RTCounter("Executable", 1);
}

Executable::~Executable() {
  if (tops_exec != nullptr) {
    RT_CHECK_NO_THROW(topsDestroyExecutable(tops_exec));
  } else if (dispatch != nullptr) {
    delete dispatch;
  }
  ResourceMgr::GetInstance()->RTCounter("ExecutableRelease", 1);
}

std::shared_ptr<Executable> Executable::Create(
    hlir::HlirDispatch *dispatch,
    const std::vector<uint64_t> input_sizes,
    const std::vector<uint64_t> output_sizes) {
  return std::make_shared<Executable>(dispatch, input_sizes, output_sizes);
}

std::shared_ptr<Executable> Executable::Create(
    const std::tuple<std::unique_ptr<char[]>, size_t> &exec) {
  topsExecutable_t gcu_executable;
  RT_CHECK(topsCreateExecutable(
      &gcu_executable, std::get<0>(exec).get(), std::get<1>(exec)));
  return std::make_shared<Executable>(gcu_executable);
}

uint64_t Executable::NumOfInputs() const {
  uint64_t exec_input_count = 0;
  if (dispatch != nullptr) {
    exec_input_count = input_sizes.size();
  } else {
    RT_CHECK(topsExecutableQueryInfo(
        tops_exec, topsExecutableInfoInputCount, &exec_input_count));
  }
  return exec_input_count;
}

uint64_t Executable::NumOfOutputs() const {
  uint64_t exec_output_count = 0;
  if (dispatch != nullptr) {
    exec_output_count = output_sizes.size();
  } else {
    RT_CHECK(topsExecutableQueryInfo(
        tops_exec, topsExecutableInfoOutputCount, &exec_output_count));
  }
  return exec_output_count;
}

void Executable::OutputSizeList(uint64_t *output_size_list) const {
  if (dispatch != nullptr) {
    for (size_t i = 0; i < output_sizes.size(); ++i) {
      output_size_list[i] = output_sizes.at(i);
    }
  } else {
    RT_CHECK(topsExecutableQueryInfo(
        tops_exec, topsExecutableInfoOutputSizeList, output_size_list));
  }
}

void Executable::InputSizeList(uint64_t *input_size_list) const {
  if (dispatch != nullptr) {
    for (size_t i = 0; i < input_sizes.size(); ++i) {
      input_size_list[i] = input_sizes.at(i);
    }
  } else {
    RT_CHECK(topsExecutableQueryInfo(
        tops_exec, topsExecutableInfoInputSizeList, input_size_list));
  }
}

}  // namespace runtime
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
