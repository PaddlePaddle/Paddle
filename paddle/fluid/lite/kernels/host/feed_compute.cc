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

#include <Eigen/Core>
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

class FeedCompute : public OpKernel<TARGET(kHost), PRECISION(kFloat)> {
 public:
  using param_t = operators::FeedParam;

  void Run() override {
    auto &theparam = param<operators::FeedParam>();
    const Tensor &feed_item = theparam.feed_list->at(theparam.col);
    theparam.out->CopyDataFrom(feed_item);
  }
};

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(feed, kHost, kFloat,
                     paddle::lite::kernels::host::FeedCompute)
    .BindInput("X", {paddle::lite::Type::Get<paddle::lite::TensorFp32NCHWTy>(
                        TARGET(kHost))})
    .BindOutput("Out", {paddle::lite::Type::Get<paddle::lite::TensorFp32NCHWTy>(
                           TARGET(kHost))})
    .Finalize();
