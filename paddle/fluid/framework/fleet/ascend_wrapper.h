/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <glog/logging.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/timer.h"

#include "ge/ge_api.h"
#include "ge/ge_api_types.h"
#include "graph/attr_value.h"
#include "graph/tensor.h"
#include "graph/types.h"

namespace paddle {
namespace framework {

// typedef std::vector<std::string> AscendGraphDesc;
typedef ge::Graph AscendGraphDesc;

class AscendInstance {
 public:
  virtual ~AscendInstance() {}
  AscendInstance() {}
  // need to expose pybind function

  std::map<std::string, std::string> GetDefaultInitOptions() {
    std::map<std::string, std::string> init_options;
    init_options["ge.exec.deviceId"] = std::to_string(0);
    init_options["ge.graphRunMode"] = std::to_string(1);
    return init_options;
  }

  std::map<std::string, std::string> GetDefaultInitSessionOptions() {
    std::map<std::string, std::string> init_options;
    init_options["a"] = "b";
    init_options["ge.trainFlag"] = "1";
    return init_options;
  }

  void InitGlobalResouces() {
    VLOG(0) << "Begin InitGlobalResouces";
    // ge::Status status = ge::GEInitialize(GetDefaultInitOptions());
    // PADDLE_ENFORCE_EQ(status, ge::SUCCESS,
    // paddle::platform::errors::PreconditionNotMet("Initialize ge failed"));

    ss = new ge::Session(GetDefaultInitSessionOptions());
    VLOG(0) << "End InitGlobalResouces";
  }

  static std::shared_ptr<AscendInstance> GetInstance() {
    if (nullptr == ascend_instance_) {
      VLOG(0) << "Initialize AscendInstance";
      ascend_instance_.reset(new paddle::framework::AscendInstance());
    }
    return ascend_instance_;
  }

  void AddAscendSubgraph(int graph_idx, const AscendGraphDesc &graph) {
    // ascend_graphs_.emplace_back(graph);
    ge::Status status = ss->AddGraph(graph_idx, graph);
    PADDLE_ENFORCE_EQ(
        status, ge::SUCCESS,
        paddle::platform::errors::PreconditionNotMet("AddGraph failed"));
  }

  ge::DataType ge_type(proto::VarType::Type type) {
    if (type == proto::VarType::FP32) {
      return ge::DataType::DT_FLOAT;
    } else if (type == proto::VarType::FP64) {
      return ge::DataType::DT_DOUBLE;
    } else if (type == proto::VarType::INT32) {
      return ge::DataType::DT_INT32;
    } else if (type == proto::VarType::INT64) {
      return ge::DataType::DT_INT64;
    } else {
      PADDLE_THROW("unsupported ge type");
    }
  }
  int ge_size(proto::VarType::Type type) {
    if (type == proto::VarType::FP32) {
      return 4;
    } else if (type == proto::VarType::FP64) {
      return 8;
    } else if (type == proto::VarType::INT32) {
      return 4;
    } else if (type == proto::VarType::INT64) {
      return 8;
    } else {
      PADDLE_THROW("unsupported ge type");
    }
  }
  ge::Tensor make_ge_tensor(const Tensor *tensor) {
    auto numel = tensor->numel();

    std::vector<int64_t> vec_dim;
    auto dimen = arity(tensor->dims());
    for (auto i = 0; i < dimen; ++i) {
      vec_dim.push_back(tensor->dims()[i]);
    }

    VLOG(0) << "input numel: " << numel << ", dimen is " << vec_dim.size()
            << ", and shape is";
    for (const auto e : vec_dim) {
      VLOG(0) << e;
    }

    ge::Shape shape(vec_dim);
    ge::TensorDesc tensor_desc(shape, ge::Format::FORMAT_ND,
                               ge_type(tensor->type()));
    tensor_desc.SetRealDimCnt(vec_dim.size());

    const uint8_t *data =
        reinterpret_cast<const uint8_t *>(tensor->data<void>());
    std::vector<uint8_t> d(numel * ge_size(tensor->type()));
    memcpy(d.data(), data,
           ge_size(tensor->type()) *
               numel);  // Note, other than 32bit may have problem
    ge::Tensor ge_tensor(tensor_desc, d);
    return ge_tensor;
  }

  void RunAscendSubgraph(int graph_idx,
                         const std::vector<const Tensor *> &inputs,
                         std::vector<Tensor *> *outputs) {
    VLOG(0) << "Ascend Graph[" << graph_idx << "] begin to run ";
    // change paddle Tensor to GE Tensor
    std::vector<ge::Tensor> ge_inputs;
    for (const auto &e : inputs) {
      ge_inputs.push_back(make_ge_tensor(e));
    }
    VLOG(0) << "construct ge tensor done.";

    // Run Graph
    std::vector<ge::Tensor> ge_outputs;
    ge::Status status = ss->RunGraph(graph_idx, ge_inputs, ge_outputs);
    PADDLE_ENFORCE_EQ(
        status, ge::SUCCESS,
        paddle::platform::errors::PreconditionNotMet("RunGraph failed"));
    VLOG(0) << "run graph done";

    // change tensor back

    for (size_t i = 0; i < ge_outputs.size(); ++i) {
      const uint8_t *ret_data = ge_outputs[i].GetData();
      size_t size = ge_outputs[i].GetSize();
      VLOG(0) << "GE Tensor size for output var " << i << " is " << size;
      auto *d = (*outputs)[i]->mutable_data<uint8_t>({static_cast<int64_t>(size)},
                                                     platform::CPUPlace());
      memcpy(d, ret_data, size);
      // for (size_t i = 0; i < size; ++i) {
      //   d[i] = ret_data[i];
      // }

      // Following for debug:
      VLOG(0) << "output for " << i << " var: ";
      float *tmp = reinterpret_cast<float *>(d);
      for (size_t j = 0; j < size / 4; ++j) {
        printf("%f ", tmp[j]);
      }
      printf("\n");
    }
  }

 protected:
  ge::Session *ss = nullptr;

 private:
  static std::shared_ptr<AscendInstance> ascend_instance_;
};
}  // end namespace framework
}  // end namespace paddle
