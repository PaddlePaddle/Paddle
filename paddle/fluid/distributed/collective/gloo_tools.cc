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
#include "paddle/fluid/distributed/collective/gloo_tools.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

void send_recv(SendRecvOptions* opts) {
  const auto& context = opts->context;
  gloo::transport::UnboundBuffer* in = opts->in.get();
  gloo::transport::UnboundBuffer* out = opts->out.get();
  const auto slot = gloo::Slot::build(kSendRecvSlotPrefix, opts->tag);

  if (context->rank == opts->src) {
    in->send(opts->dst, slot);
    in->waitSend(opts->timeout);
  } else if (context->rank == opts->dst) {
    out->recv(opts->src, slot);
    out->waitRecv(opts->timeout);
  }
}

void gather(GatherCommOptions* opts) {
  const auto& context = opts->context;
  gloo::transport::UnboundBuffer* in = opts->in.get();
  gloo::transport::UnboundBuffer* out = opts->out.get();
  const auto slot = gloo::Slot::build(gloo::kGatherSlotPrefix, opts->tag);

  PADDLE_ENFORCE_GE(opts->root,
                    0,
                    platform::errors::InvalidArgument(
                        "root must be greater and equal than 0."));
  PADDLE_ENFORCE_LT(
      opts->root,
      context->size,
      platform::errors::InvalidArgument("root must be less than worldsize."));
  PADDLE_ENFORCE_NE(
      in,
      nullptr,
      platform::errors::InvalidArgument("input buffer must not be nullptr."));
  PADDLE_ENFORCE_GT(
      opts->elementSize,
      0,
      platform::errors::InvalidArgument("elementSize must be greater than 0."));

  if (context->rank == opts->root) {
    PADDLE_ENFORCE_NE(out,
                      nullptr,
                      platform::errors::InvalidArgument(
                          "output buffer must not be nullptr."));
    PADDLE_ENFORCE_GE(out->size,
                      in->size * context->size,
                      platform::errors::InvalidArgument(
                          "output buffer size must be greater than input "
                          "buffer size * worldsize."));
    memcpy(static_cast<uint8_t*>(out->ptr) + context->rank * in->size,
           static_cast<uint8_t*>(in->ptr),
           in->size);
  }

  if (context->rank == opts->root) {
    size_t numRecv = 0;
    for (int64_t rank = 0; rank < context->size; rank++) {
      if (rank == context->rank) {
        continue;
      }

      size_t recvOffset = rank * in->size;
      out->recv(rank, slot, recvOffset, in->size);
      numRecv++;
    }
    for (size_t i = 0; i < numRecv; i++) {
      out->waitRecv(opts->timeout);
    }
  } else {
    in->send(opts->root, slot);
    in->waitSend(opts->timeout);
  }
}

}  // namespace distributed
}  // namespace paddle
