/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

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
#include "paddle/pserver/ParameterClient.h"
#include "paddle/pserver/ParameterServer.h"
#include "paddle/parameter/Parameter.h"
#include <Python.h>

namespace paddle {

struct PyObjectDeleter {
  void operator()(PyObject* obj) {
    if (obj) {
      Py_DECREF(obj);
    }
  }
};

class ParameterClientPy : public ParameterClient {
protected:
  typedef std::unique_ptr<PyObject, PyObjectDeleter> PyObjectPtr;

  std::vector<ParameterPtr> parameter_;
  int initArgc_;
  char** initArgv_;

public:
  ParameterClientPy(std::vector<std::string> configs, int argc,
                    std::vector<std::string> argv, bool useGpu) {
    initArgc_ = argc;
    initArgv_ = new char* [argc];
    for (int i = 0; i < argc; i++) {
      initArgv_[i] = new char[argv[i].size()];
      strcpy(initArgv_[i],      // NOLINT
             argv[i].c_str());  // NOLINT TODO(yuyang18): use snprintf instead.
    }
    ParameterConfig pyConfig;
    ParameterPtr param;
    for (auto& config : configs) {
      pyConfig.ParseFromString(config);
      param.reset(new Parameter(pyConfig, useGpu));
      parameter_.push_back(param);
    }
    Py_Initialize();
    CHECK(Py_IsInitialized());
  }

  ~ParameterClientPy() {
    delete initArgv_;
    Py_Finalize();
  }

  Parameter getParameter(int idx) { return *(parameter_[idx].get()); }

  void initClientPy() {
    initMain(initArgc_, initArgv_);
    CHECK(init(parameter_)) << "Init Client Failed.";
  }

  void setConfigPy(std::string config) {
    OptimizationConfig optConfig;
    optConfig.ParseFromString(config);
    setConfig(optConfig);
  }

  bool inStatusPy(int status) { return inStatus(PServerStatus(status)); }

  void setStatusPy(int status) { setStatus(PServerStatus(status)); }

  void waitForStatusPy(int status) { waitForStatus(PServerStatus(status)); }

  void sendParameterPy(int updateMode, int parameterType, int numSamples,
                       real cost, bool sendBackParameter) {
    sendParameter(ParameterUpdateMode(updateMode), ParameterType(parameterType),
                  int64_t(numSamples), real(cost), sendBackParameter);
  }

  template <class ProtoIn, class ProtoOut>
  std::string asyncCallPy(const char* serviceName, const char* funcName,
                          const std::string in) {
    ProtoIn protoIn;
    ProtoOut protoOut;
    std::mutex waitLock;
    std::string data;
    protoIn.ParseFromString(in);
    waitLock.lock();
    auto callback = [&](ProtoOut* pOut, bool isSuccessful) {
      if (isSuccessful) {
        pOut->SerializeToString(&data);
      } else {
        LOG(INFO) << "Async Talk Failed.";
      }
      waitLock.unlock();
    };

    ubClient_.asyncCall<ProtoIn, ProtoOut>(serviceName, funcName, protoIn,
                                           &protoOut, callback);
    waitLock.lock();
    protoOut.SerializeToString(&data);
    return data;
  }
};

}  // namespace paddle
