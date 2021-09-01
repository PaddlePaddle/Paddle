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
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

namespace paddle {
namespace platform {

void ParseCommandLineFlags(int argc, char** argv, bool remove);

}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace framework {

bool InitGflags(std::vector<std::string> argv);

void InitGLOG(const std::string& prog_name);

void InitDevices();

void InitDevices(const std::vector<int> devices);

#ifndef _WIN32
class SignalMessageDumper {
 public:
  ~SignalMessageDumper() {}
  SignalMessageDumper(const SignalMessageDumper& o) = delete;
  const SignalMessageDumper& operator=(const SignalMessageDumper& o) = delete;

  static SignalMessageDumper& Instance() {
    static SignalMessageDumper instance;
    return instance;
  }

  std::shared_ptr<std::ostringstream> Get() { return dumper_; }

 private:
  SignalMessageDumper() : dumper_(new std::ostringstream()) {}
  std::shared_ptr<std::ostringstream> dumper_;
};

void SignalHandle(const char* data, int size);
#endif

void DisableSignalHandler();

}  // namespace framework
}  // namespace paddle
