// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/fleet_executor/amplifier_interceptor.h"

#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace distributed {

AmplifierInterceptor::AmplifierInterceptor(int64_t interceptor_id,
                                           TaskNode* node)
    : ComputeInterceptor(interceptor_id, node) {
  run_per_steps_ = node->run_per_steps();
  run_at_offset_ = node->run_at_offset();
  reply_up_per_steps_ = node->reply_up_per_steps();
  send_down_per_steps_ = node->send_down_per_steps();

  PADDLE_ENFORCE_GE(
      run_per_steps_, 1,
      platform::errors::InvalidArgument(
          "run_per_steps must >= 1, but now is %ld", run_per_steps_));
  PADDLE_ENFORCE_GE(
      run_at_offset_, 0,
      platform::errors::InvalidArgument(
          "run_at_offset must >= 0, but now is %ld", run_at_offset_));
  PADDLE_ENFORCE_LT(run_at_offset_, run_per_steps_,
                    platform::errors::InvalidArgument(
                        "run_at_offset must < run_per_steps, must now "
                        "run_at_offset=%ld run_per_steps=%ld",
                        run_at_offset_, run_per_steps_));
  PADDLE_ENFORCE_GE(
      reply_up_per_steps_, 1,
      platform::errors::InvalidArgument(
          "reply_up_per_steps must >= 1, but now is %ld", reply_up_per_steps_));
  PADDLE_ENFORCE_GE(send_down_per_steps_, 1,
                    platform::errors::InvalidArgument(
                        "send_down_per_steps must >= 1, but now is %ld",
                        send_down_per_steps_));
}

void AmplifierInterceptor::RunOps() {
  // run_per_steps_, run_at_offset_
  // 4, 0 --> run at step 0, 4, 8, 12
  // 4, 3 --> run at step 3, 7, 11, 15
  if ((step_ % run_per_steps_) == run_at_offset_) {
    ComputeInterceptor::RunOps();
  }
}

void AmplifierInterceptor::SendDataReadyToDownStream() {
  // run multi times, send ready one times to downstream, that is
  // input multi times, output one times
  if (step_ % send_down_per_steps_ == 0) {
    ComputeInterceptor::SendDataReadyToDownStream();
  }
}

void AmplifierInterceptor::ReplyCompletedToUpStream() {
  // run multi times, reply one times to upstream, that is
  // input one times, output multi times
  if (step_ % reply_up_per_steps_ == 0) {
    ComputeInterceptor::ReplyCompletedToUpStream();
  }
}

REGISTER_INTERCEPTOR(Amplifier, AmplifierInterceptor);

}  // namespace distributed
}  // namespace paddle
