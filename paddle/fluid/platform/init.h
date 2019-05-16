/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <mutex>  // NOLINT
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

namespace paddle {
namespace framework {

void InitGflags(std::vector<std::string> argv);

void InitGLOG(const std::string &prog_name);

void InitDevices(bool init_p2p);

void InitDevices(bool init_p2p, const std::vector<int> devices);

void InitDGC();

}  // namespace framework
}  // namespace paddle
