// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/basic_engine.h"
#include "paddle/fluid/imperative/hooks.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/memory/memcpy.h"

namespace platform = paddle::platform;
namespace framework = paddle::framework;
namespace memory = paddle::memory;

namespace paddle {
namespace imperative {

using vb_vector = std::vector<std::shared_ptr<imperative::VarBase>>;
using var_pair = std::pair<std::string, vb_vector>;

TEST(TestHooks, TestGradAccumulatorPostHook) {
  // 1. prepare
  Tracer tracer;
  std::shared_ptr<VarBase> x(new VarBase(true, "x"));
  std::shared_ptr<VarBase> y(new VarBase(true, "y"));
  std::shared_ptr<VarBase> out(new VarBase(true, "out"));
  x->SetOverridedStopGradient(false);
  y->SetOverridedStopGradient(false);

  platform::CPUPlace place;
  std::vector<float> src_data(10, 2.0);
  std::vector<int64_t> x_dims = {2, 5};
  std::vector<int64_t> y_dims = {5, 2};

  auto* x_tensor = x->MutableVar()->GetMutable<framework::LoDTensor>();
  auto* y_tensor = y->MutableVar()->GetMutable<framework::LoDTensor>();

  x_tensor->Resize(framework::make_ddim(x_dims));
  auto* mutable_x = x_tensor->mutable_data<float>(place);
  memory::Copy(place, mutable_x, place, src_data.data(),
               sizeof(float) * src_data.size());

  y_tensor->Resize(framework::make_ddim(y_dims));
  auto* mutable_y = y_tensor->mutable_data<float>(place);
  memory::Copy(place, mutable_y, place, src_data.data(),
               sizeof(float) * src_data.size());

  var_pair x_pair = var_pair("X", vb_vector(1, x));
  var_pair y_pair = var_pair("Y", vb_vector(1, y));
  var_pair out_pair = var_pair("Out", vb_vector(1, out));

  NameVarBaseMap ins = {x_pair, y_pair};
  NameVarBaseMap outs = {out_pair};
  framework::AttributeMap mul_attr_map;
  mul_attr_map["use_mkldnn"] = false;

  // add GradAccumulatorPostHook
  auto x_var_wrapper = x->SharedVar();
  x_var_wrapper->SetGradReduceHook(
      new LambdaGradAccumulatorPostHook([=](VariableWrapper* grad) {
        auto* grad_tensor =
            grad->MutableVar()->GetMutable<framework::LoDTensor>();
        for (int i = 0; i < grad_tensor->numel(); ++i) {
          grad_tensor->mutable_data<float>(place)[i] *= 2.0;
        }
      }));

  // 2. forward
  VLOG(0) << "trace op forward.";
  tracer.TraceOp("mul", ins, outs, mul_attr_map, place, true);

  ASSERT_EQ(x->GradVarBase()->GradOpNum(), 0UL);
  ASSERT_EQ(y->GradVarBase()->GradOpNum(), 0UL);
  ASSERT_EQ(out->GradVarBase()->GradOpNum(), 1UL);

  // 3. backward
  BasicEngine engine;
  engine.Init(out.get());
  engine.Execute();

  framework::LoDTensor x_grad;
  framework::TensorCopySync(x->GradVar().Get<framework::LoDTensor>(), place,
                            &x_grad);
  for (int i = 0; i < x_grad.numel(); ++i) {
    ASSERT_EQ(x_grad.data<float>()[i], 8.0);
  }

  framework::LoDTensor y_grad;
  framework::TensorCopySync(y->GradVar().Get<framework::LoDTensor>(), place,
                            &y_grad);

  for (int i = 0; i < y_grad.numel(); ++i) {
    ASSERT_EQ(y_grad.data<float>()[i], 4.0);
  }
}

}  // namespace imperative
}  // namespace paddle

USE_OP(mul);
USE_OP(mul_grad);
