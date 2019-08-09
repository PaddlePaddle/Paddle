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

#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

class FeedCompute
    : public KernelLite<TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  using param_t = operators::FeedParam;

  void Run() override {
    auto &param = Param<operators::FeedParam>();
    VLOG(4) << "feed_list.size: " << param.feed_list->size();
    VLOG(4) << "col " << param.col;
    const lite::Tensor &feed_item = (*param.feed_list)[param.col];
    param.out->ShareDataWith(feed_item);
  }
};

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(feed, kHost, kAny, kAny,
                     paddle::lite::kernels::host::FeedCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
