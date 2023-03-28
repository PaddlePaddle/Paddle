// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <cstring>

#include "gloo/common/logging.h"
#include "gloo/math.h"
#include "gloo/types.h"
#include "paddle/fluid/distributed/collective/send_recv.h"

namespace paddle {
namespace distributed {

void send_recv(SendRecvOptions* opts) {
  const auto& context = opts->context;
  gloo::transport::UnboundBuffer* in = opts->in.get();
  gloo::transport::UnboundBuffer* out = opts->out.get();
  const auto slot = gloo::Slot::build(kSendRecvSlotPrefix, opts->tag);

  if (context->rank == opts->src) {
    in->send((context->rank + 1) % 2, slot);
    in->waitSend(opts->timeout);
  } else {
    out->recv((context->rank + 1) % 2, slot);
    out->waitRecv(opts->timeout);
  }
}

}  // namespace distributed
}  // namespace paddle
