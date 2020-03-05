/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/stream/paddle_stream.h"
#include "paddle/fluid/framework/parallel_executor.h"

namespace paddle {
namespace platform {
namespace stream {

BaseStream::BaseStream(framework::ParallelExecutor* pe)
    : allocated_(false),
      ok_(false),
      implementation_(pe->CreateStreamImplementation()),
      pe_(pe) {}

BaseStream::~BaseStream() {}

BaseStream& BaseStream::Init() {
  if (pe_) {
    pe_->AllocateStream(this);
    allocated_ = true;
    ok_ = true;
  }
  return *this;
}

BaseStream& BaseStream::WaitForOtherStream(BaseStream* other) {
  PADDLE_ENFORCE_NE(this, other, "should wait on different stream");
  PADDLE_ENFORCE_NE(pe_, nullptr, "PE should not be nullptr");
  PADDLE_ENFORCE_EQ(pe_->CreateStreamDependency(this, other), true,
                    "wait dependency should be ok");
  return *this;
}

}  // namespace stream
}  // namespace platform
}  // namespace paddle
