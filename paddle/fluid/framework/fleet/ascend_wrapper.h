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

#ifdef PADDLE_WITH_ASCEND_CL
#include <glog/logging.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/timer.h"

#include "ge/ge_api.h"
#include "graph/attr_value.h"
#include "graph/tensor.h"
#include "graph/types.h"

namespace paddle {
namespace framework {

typedef ge::Graph AscendGraphDesc;

#ifdef PADDLE_WITH_ASCEND_STRING
using AscendString = ge::AscendString;
#else
using AscendString = std::string;
#endif

class AscendInstance {
 public:
  virtual ~AscendInstance() {}
  AscendInstance() {}

  std::map<AscendString, AscendString> _GetDefaultInitOptions() {
    std::map<AscendString, AscendString> init_options;
    init_options["ge.exec.deviceId"] = "0";
    init_options["ge.graphRunMode"] = "1";
    return init_options;
  }

  std::map<AscendString, AscendString> _GetDefaultInitSessionOptions() {
    std::map<AscendString, AscendString> init_options;
    // init_options["a"] = "b";
    // init_options["ge.trainFlag"] = "1";
    return init_options;
  }

  ge::Status InitGEForUT() {
    return ge::GEInitialize(_GetDefaultInitOptions());
  }

  void InitGlobalResouces() {
    LOG(INFO) << "Begin ascend InitGlobalResouces";
    session_.reset(new ge::Session(_GetDefaultInitSessionOptions()));
    if (session_ == nullptr) {
      PADDLE_THROW(platform::errors::Fatal("new session error: nullptr"));
    }
    LOG(INFO) << "End ascend InitGlobalResouces";
  }

  void DestroyGlobalResouces() {
    LOG(INFO) << "Begin ascend DestroyGlobalResouces";
    session_ = nullptr;
    LOG(INFO) << "Begin ascend DestroyGlobalResouces";
  }

  static std::shared_ptr<AscendInstance> GetInstance() {
    if (nullptr == ascend_instance_) {
      ascend_instance_.reset(new paddle::framework::AscendInstance());
      VLOG(1) << "Initialize AscendInstance Done";
    }
    return ascend_instance_;
  }

  void AddAscendSubgraph(int graph_idx, const AscendGraphDesc &graph) {
    ge::Status status = session_->AddGraph(graph_idx, graph);
    PADDLE_ENFORCE_EQ(status, ge::SUCCESS,
                      paddle::platform::errors::PreconditionNotMet(
                          "Calling addGraph of graph engine failed, please "
                          "check Ascend Log."));
    VLOG(1) << "AddAscendSubgraph " << graph_idx << " Done";
  }

  ge::DataType VarTypeToGeType(proto::VarType::Type type) {
    if (type == proto::VarType::FP16) {
      return ge::DataType::DT_FLOAT16;
    } else if (type == proto::VarType::FP32) {
      return ge::DataType::DT_FLOAT;
    } else if (type == proto::VarType::FP64) {
      return ge::DataType::DT_DOUBLE;
    } else if (type == proto::VarType::INT32) {
      return ge::DataType::DT_INT32;
    } else if (type == proto::VarType::INT64) {
      return ge::DataType::DT_INT64;
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Not support %s as tensor type.", DataTypeToString(type)));
    }
  }
  int GeTypeSize(proto::VarType::Type type) {
    if (type == proto::VarType::FP16) {
      return 2;
    } else if (type == proto::VarType::FP32) {
      return 4;
    } else if (type == proto::VarType::FP64) {
      return 8;
    } else if (type == proto::VarType::INT32) {
      return 4;
    } else if (type == proto::VarType::INT64) {
      return 8;
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Not support %s as tensor type.", DataTypeToString(type)));
    }
  }
  ge::Tensor ConvertToGeTensor(const Tensor *tensor) {
    auto numel = tensor->numel();
    std::vector<int64_t> vec_dim;
    auto dimen = arity(tensor->dims());
    for (auto i = 0; i < dimen; ++i) {
      vec_dim.push_back(tensor->dims()[i]);
    }
    // For Debug
    // VLOG(1) << "input numel: " << numel << ", dimen is " << vec_dim.size() <<
    // ", and shape is";
    // for (const auto e : vec_dim) {
    //   VLOG(0) << e;
    // }

    ge::Shape shape(vec_dim);
    ge::TensorDesc tensor_desc(
        shape, ge::Format::FORMAT_ND,
        VarTypeToGeType(framework::TransToProtoVarType(tensor->dtype())));
    tensor_desc.SetRealDimCnt(vec_dim.size());

    const uint8_t *data = reinterpret_cast<const uint8_t *>(tensor->data());
    std::vector<uint8_t> dst(
        numel * GeTypeSize(framework::TransToProtoVarType(tensor->dtype())));
    memcpy(dst.data(), data,
           GeTypeSize(framework::TransToProtoVarType(tensor->dtype())) * numel);
    ge::Tensor ge_tensor(tensor_desc, dst);
    return ge_tensor;
  }

  void RunAscendSubgraph(int graph_idx,
                         const std::vector<const Tensor *> &inputs,
                         std::vector<Tensor *> *outputs) {
    VLOG(1) << "Ascend Graph[" << graph_idx << "] is about to run.";
    // Convert paddle Tensor to GE Tensor
    std::vector<ge::Tensor> ge_inputs;
    for (const auto &e : inputs) {
      ge_inputs.push_back(ConvertToGeTensor(e));
    }

    // Run Graph
    std::vector<ge::Tensor> ge_outputs;
    ge::Status status = session_->RunGraph(graph_idx, ge_inputs, ge_outputs);
    PADDLE_ENFORCE_EQ(status, ge::SUCCESS,
                      paddle::platform::errors::PreconditionNotMet(
                          "Calling RunGraph of graph engine failed, please "
                          "check Ascend Log."));
    VLOG(1) << "Run Ascend Graph[" << graph_idx << "] Done";

    // change tensor back, note all tensor's type computed in GE is uint8
    for (size_t i = 0; i < ge_outputs.size(); ++i) {
      const uint8_t *ret_data = ge_outputs[i].GetData();
      size_t size = ge_outputs[i].GetSize();
      VLOG(1) << "GE Tensor size of the " << i << "th output var is " << size;
      auto *dst = (*outputs)[i]->mutable_data<uint8_t>({(int64_t)size},
                                                       platform::CPUPlace());
      memcpy(dst, ret_data, size);

      // Following for debug:
      // VLOG(0) << "output for " << i << " var: ";
      // float *tmp = reinterpret_cast<float*>(dst);
      // for (size_t j = 0; j < size / 4; ++j) {
      //   printf("%f ", tmp[j]);
      // }
      // printf("\n");
    }
  }

 protected:
  std::shared_ptr<ge::Session> session_;

 private:
  static std::shared_ptr<AscendInstance> ascend_instance_;
};
}  // namespace framework
}  // namespace paddle
#endif
