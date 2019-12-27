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
  for (int i = 0; i < vout->Var().Get<framework::LoDTensor>().numel(); i++) {
    ASSERT_EQ(out_tensor.data<float>()[i], 20.0);
  }
}

TEST(test_tracer, test_trace_op_with_backward) {
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
  for (int i = 0; i < vout->Var().Get<framework::LoDTensor>().numel(); i++) {
    ASSERT_EQ(out_tensor.data<float>()[i], 20.0);
  }
}

TEST(test_tracer, test_track_backward_output) {
  // Doing an mul
  imperative::Tracer tracer;
  std::shared_ptr<imperative::VarBase> x_in(
      new imperative::VarBase(true, "x_in"));
  std::shared_ptr<imperative::VarBase> y_in(
      new imperative::VarBase(true, "y_in"));
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
  tracer.TraceOp("mul", ins, outs, mul_attr_map, place, true);
  auto* engine = tracer.GetDefaultEngine();
  ASSERT_NE(engine->GradVars().size(), 0UL);
  ASSERT_NE(engine->GradOps().size(), 0UL);  // trace_backward already ran.
}

TEST(test_tracer, test_track_backward_input) {
  // Doing an mul
  imperative::Tracer tracer;
  std::shared_ptr<imperative::VarBase> x_in(
      new imperative::VarBase(true, "x_in"));
  std::shared_ptr<imperative::VarBase> y_in(
      new imperative::VarBase(true, "y_in"));
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(true, "vout"));
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
  tracer.TraceOp("mul", ins, outs, mul_attr_map, place, true);
  auto* engine = tracer.GetDefaultEngine();
  ASSERT_NE(engine->GradVars().size(), 0UL);
  ASSERT_NE(engine->GradOps().size(), 0UL);  // trace_backward already ran.
}
#if defined(PADDLE_WITH_CUDA)
TEST(test_tracer, test_trace_op_with_multi_device_inputs) {
  // Doing an mul
  imperative::Tracer tracer;
  std::shared_ptr<imperative::VarBase> x_in(
      new imperative::VarBase(true, "x_in"));
  x_in->SetOverridedStopGradient(false);  // force to run backward
  std::shared_ptr<imperative::VarBase> y_in(
      new imperative::VarBase(true, "y_in"));
  y_in->SetOverridedStopGradient(false);
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(true, "vout"));
  platform::CPUPlace place;
  platform::CUDAPlace gpu_place(0);
  std::vector<float> src_data(10, 2.0);
  std::vector<int64_t> dims1 = {2, 5};
  std::vector<int64_t> dims2 = {2, 5};

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
  tracer.TraceOp("elementwise_add", ins, outs, mul_attr_map, gpu_place, true);

  // run reduce sum
  std::shared_ptr<imperative::VarBase> reduce_sum_out(
      new imperative::VarBase(true, "reduce_sum_out"));
  var_pair reduce_sum_in_pair = var_pair("X", vb_vector(1, vout));
  var_pair reduce_sum_out_pair = var_pair("Out", vb_vector(1, reduce_sum_out));
  imperative::NameVarBaseMap reduce_in = {reduce_sum_in_pair};
  imperative::NameVarBaseMap reduce_out = {reduce_sum_out_pair};
  framework::AttributeMap reduce_attr_map;
  tracer.TraceOp("reduce_sum", reduce_in, reduce_out, reduce_attr_map,
                 gpu_place, true);
  detail::BackwardStrategy back_st;
  imperative::Engine* engine = tracer.GetDefaultEngine();
  engine->Init(reduce_sum_out.get(), back_st);
  engine->Execute();

  framework::LoDTensor rlt;
  framework::TensorCopySync(vout->Var().Get<framework::LoDTensor>(), place,
                            &rlt);
  for (int i = 0; i < rlt.numel(); i++) {
    ASSERT_EQ(rlt.data<float>()[i], 4.0);
  }

  framework::LoDTensor out_grad;
  framework::TensorCopySync(vout->GradVar().Get<framework::LoDTensor>(), place,
                            &out_grad);
  for (int i = 0; i < out_grad.numel(); ++i) {
    ASSERT_EQ(out_grad.data<float>()[i], 1.0);
  }

  framework::LoDTensor x_grad;
  framework::TensorCopySync(x_in->GradVar().Get<framework::LoDTensor>(), place,
                            &x_grad);

  for (int i = 0; i < x_grad.numel(); ++i) {
    ASSERT_EQ(x_grad.data<float>()[i], 1.0);
  }

  framework::LoDTensor y_grad;
  framework::TensorCopySync(y_in->GradVar().Get<framework::LoDTensor>(), place,
                            &y_grad);

  for (int i = 0; i < y_grad.numel(); ++i) {
    ASSERT_EQ(y_grad.data<float>()[i], 1.0);
  }
}

#endif

TEST(test_tracer, test_unique_name_generator) {
  // generate two unique names
  imperative::Tracer tracer;
  auto fc_1 = tracer.GenerateUniqueName("fc");
  auto fc_2 = tracer.GenerateUniqueName("fc");
  ASSERT_STREQ("fc_0", fc_1.c_str());
  ASSERT_STREQ("fc_1", fc_2.c_str());
}

TEST(test_tracer, test_current_tracer) {
  // use current_tracer
  auto tracer = std::make_shared<imperative::Tracer>();
  imperative::SetCurrentTracer(tracer);
  auto current_tracer = imperative::GetCurrentTracer();
  ASSERT_EQ(current_tracer, tracer);
}

TEST(test_tracer, test_expected_place) {
  // default expected place is CPUPlace
  imperative::Tracer tracer;
  ASSERT_EQ(platform::is_cpu_place(tracer.ExpectedPlace()), true);
  // set to CUDAPlace
  platform::CUDAPlace gpu_place(0);
  tracer.SetExpectedPlace(gpu_place);
  ASSERT_EQ(platform::is_gpu_place(tracer.ExpectedPlace()), true);
}

TEST(test_tracer, test_var_without_grad_var) {
  // Doing an mul
  imperative::Tracer tracer;
  std::shared_ptr<imperative::VarBase> x_in(
      new imperative::VarBase(true, "x_in"));
  x_in->ClearGradVarBase();
  std::shared_ptr<imperative::VarBase> y_in(
      new imperative::VarBase(true, "y_in"));
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(true, "vout"));
  x_in->SetOverridedStopGradient(false);
  y_in->SetOverridedStopGradient(false);
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
  for (int i = 0; i < vout->Var().Get<framework::LoDTensor>().numel(); i++) {
    ASSERT_EQ(out_tensor.data<float>()[i], 20.0);
  }

  detail::BackwardStrategy back_st;
  imperative::Engine* engine = tracer.GetDefaultEngine();
  ASSERT_NE(engine->GradVars().size(), 0UL);
  ASSERT_NE(engine->GradOps().size(), 0UL);  // trace_backward already ran.
  engine->Init(vout.get(), back_st);
  engine->Execute();

  // check the grad
  framework::LoDTensor x_grad;
  framework::TensorCopySync(x_in->GradVar().Get<framework::LoDTensor>(), place,
                            &x_grad);

  for (int i = 0; i < x_grad.numel(); ++i) {
    ASSERT_EQ(x_grad.data<float>()[i], 4.0);
  }

  framework::LoDTensor y_grad;
  framework::TensorCopySync(y_in->GradVar().Get<framework::LoDTensor>(), place,
                            &y_grad);

  for (int i = 0; i < y_grad.numel(); ++i) {
    ASSERT_EQ(y_grad.data<float>()[i], 4.0);
  }
}

}  // namespace imperative
}  // namespace paddle

USE_OP(mul);
USE_OP(reduce_sum);
USE_OP(reduce_sum_grad);
USE_OP(elementwise_add);
