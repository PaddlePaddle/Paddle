// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

//
// Created by Jiabin on 2019-08-16.
//

#include <paddle/fluid/framework/op_registry.h>
#include <memory>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/memory/memcpy.h"

namespace imperative = paddle::imperative;
namespace platform = paddle::platform;
namespace framework = paddle::framework;

namespace paddle {
namespace imperative {

using vb_vector = std::vector<std::shared_ptr<imperative::VarBase>>;

using var_pair = std::pair<std::string, vb_vector>;

TEST(test_tracer, test_trace_op) {
  // Doing an mul
  imperative::Tracer tracer;
  std::shared_ptr<imperative::VarBase> x_in(
      new imperative::VarBase(true, "x_in"));
  std::shared_ptr<imperative::VarBase> y_in(
      new imperative::VarBase(true, "y_in"));
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(true, "vout"));
  platform::CPUPlace place;
  std::vector<float> src_data(10, 2.0);
  std::vector<int64_t> dims1 = {2, 5};
  std::vector<int64_t> dims2 = {5, 2};

  auto* x_in_tensor = x_in->MutableVar()->GetMutable<framework::LoDTensor>();
  auto* y_in_tensor = y_in->MutableVar()->GetMutable<framework::LoDTensor>();
  x_in_tensor->Resize(framework::make_ddim(dims1));
  auto* mutable_x = x_in_tensor->mutable_data<float>(place);
  paddle::memory::Copy(place, mutable_x, place, src_data.data(),
                       sizeof(float) * src_data.size());
  y_in_tensor->Resize(framework::make_ddim(dims2));
  auto* mutable_y = y_in_tensor->mutable_data<float>(place);
  paddle::memory::Copy(place, mutable_y, place, src_data.data(),
                       sizeof(float) * src_data.size());

  var_pair x_pair = var_pair("X", vb_vector(1, x_in));
  var_pair y_pair = var_pair("Y", vb_vector(1, y_in));
  var_pair out_pair = var_pair("Out", vb_vector(1, vout));
  imperative::NameVarBaseMap ins = {x_pair, y_pair};
  imperative::NameVarBaseMap outs = {out_pair};
  framework::AttributeMap mul_attr_map;
  mul_attr_map["use_mkldnn"] = false;
  tracer.TraceOp("mul", ins, outs, mul_attr_map, place, true);
  const auto& out_tensor = vout->Var().Get<framework::LoDTensor>();
  for (size_t i = 0; i < vout->Var().Get<framework::LoDTensor>().numel(); i++) {
    ASSERT_EQ(out_tensor.data<float>()[i], 20.0);
  }
}

TEST(test_tracer, test_track_backward_output) {
  // Doing an mul
  imperative::Tracer tracer;
  std::shared_ptr<imperative::VarBase> x_in(
      new imperative::VarBase(true, "x_in"));
  std::shared_ptr<imperative::VarBase> y_in(
      new imperative::VarBase(false, "y_in"));
  x_in->SetOverridedStopGradient(false);
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(true, "vout"));
  platform::CPUPlace place;
  std::vector<float> src_data(10, 2.0);
  std::vector<int64_t> dims1 = {2, 5};
  std::vector<int64_t> dims2 = {5, 2};

  auto* x_in_tensor = x_in->MutableVar()->GetMutable<framework::LoDTensor>();
  auto* y_in_tensor = y_in->MutableVar()->GetMutable<framework::LoDTensor>();
  x_in_tensor->Resize(framework::make_ddim(dims1));
  auto* mutable_x = x_in_tensor->mutable_data<float>(place);
  paddle::memory::Copy(place, mutable_x, place, src_data.data(),
                       sizeof(float) * src_data.size());
  y_in_tensor->Resize(framework::make_ddim(dims2));
  auto* mutable_y = y_in_tensor->mutable_data<float>(place);
  paddle::memory::Copy(place, mutable_y, place, src_data.data(),
                       sizeof(float) * src_data.size());

  var_pair x_pair = var_pair("X", vb_vector(1, x_in));
  var_pair y_pair = var_pair("Y", vb_vector(1, y_in));
  var_pair out_pair = var_pair("Out", vb_vector(1, vout));
  imperative::NameVarBaseMap ins = {x_pair, y_pair};
  imperative::NameVarBaseMap outs = {out_pair};
  framework::AttributeMap mul_attr_map;
  mul_attr_map["use_mkldnn"] = false;
  ASSERT_ANY_THROW(tracer.TraceOp("mul", ins, outs, mul_attr_map, place, true));
}

TEST(test_tracer, test_track_backward_input) {
  // Doing an mul
  imperative::Tracer tracer;
  std::shared_ptr<imperative::VarBase> x_in(
      new imperative::VarBase(true, "x_in"));
  std::shared_ptr<imperative::VarBase> y_in(
      new imperative::VarBase(true, "y_in"));
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(false, "vout"));
  platform::CPUPlace place;
  x_in->SetOverridedStopGradient(false);
  std::vector<float> src_data(10, 2.0);
  std::vector<int64_t> dims1 = {2, 5};
  std::vector<int64_t> dims2 = {5, 2};

  auto* x_in_tensor = x_in->MutableVar()->GetMutable<framework::LoDTensor>();
  auto* y_in_tensor = y_in->MutableVar()->GetMutable<framework::LoDTensor>();
  x_in_tensor->Resize(framework::make_ddim(dims1));
  auto* mutable_x = x_in_tensor->mutable_data<float>(place);
  paddle::memory::Copy(place, mutable_x, place, src_data.data(),
                       sizeof(float) * src_data.size());
  y_in_tensor->Resize(framework::make_ddim(dims2));
  auto* mutable_y = y_in_tensor->mutable_data<float>(place);
  paddle::memory::Copy(place, mutable_y, place, src_data.data(),
                       sizeof(float) * src_data.size());

  var_pair x_pair = var_pair("X", vb_vector(1, x_in));
  var_pair y_pair = var_pair("Y", vb_vector(1, y_in));
  var_pair out_pair = var_pair("Out", vb_vector(1, vout));
  imperative::NameVarBaseMap ins = {x_pair, y_pair};
  imperative::NameVarBaseMap outs = {out_pair};
  framework::AttributeMap mul_attr_map;
  mul_attr_map["use_mkldnn"] = false;
  ASSERT_ANY_THROW(tracer.TraceOp("mul", ins, outs, mul_attr_map, place, true));
}
#if defined(PADDLE_WITH_CUDA)
TEST(test_tracer, test_trace_op_with_multi_device_inputs) {
  // Doing an mul
  imperative::Tracer tracer;
  std::shared_ptr<imperative::VarBase> x_in(
      new imperative::VarBase(true, "x_in"));
  std::shared_ptr<imperative::VarBase> y_in(
      new imperative::VarBase(true, "y_in"));
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(true, "vout"));
  platform::CPUPlace place;
  platform::CUDAPlace gpu_place(0);
  std::vector<float> src_data(10, 2.0);
  std::vector<int64_t> dims1 = {2, 5};
  std::vector<int64_t> dims2 = {5, 2};

  auto* x_in_tensor = x_in->MutableVar()->GetMutable<framework::LoDTensor>();
  auto* y_in_tensor = y_in->MutableVar()->GetMutable<framework::LoDTensor>();
  x_in_tensor->Resize(framework::make_ddim(dims1));
  auto* mutable_x = x_in_tensor->mutable_data<float>(place);
  paddle::memory::Copy(place, mutable_x, place, src_data.data(),
                       sizeof(float) * src_data.size());
  y_in_tensor->Resize(framework::make_ddim(dims2));
  auto* mutable_y = y_in_tensor->mutable_data<float>(gpu_place);
  paddle::memory::Copy(gpu_place, mutable_y, place, src_data.data(),
                       sizeof(float) * src_data.size(), 0);
  var_pair x_pair = var_pair("X", vb_vector(1, x_in));
  var_pair y_pair = var_pair("Y", vb_vector(1, y_in));
  var_pair out_pair = var_pair("Out", vb_vector(1, vout));
  imperative::NameVarBaseMap ins = {x_pair, y_pair};
  imperative::NameVarBaseMap outs = {out_pair};
  framework::AttributeMap mul_attr_map;
  mul_attr_map["use_mkldnn"] = false;
  tracer.TraceOp("mul", ins, outs, mul_attr_map, gpu_place, true);
  framework::LoDTensor rlt;
  framework::TensorCopySync(vout->Var().Get<framework::LoDTensor>(), place,
                            &rlt);
  for (size_t i = 0; i < rlt.numel(); i++) {
    ASSERT_EQ(rlt.data<float>()[i], 20.0);
  }
}
#endif
}  // namespace imperative
}  // namespace paddle

USE_OP(mul);
