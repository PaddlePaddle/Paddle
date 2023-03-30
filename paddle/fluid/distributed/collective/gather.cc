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
#include "gloo/types.h"
#include "paddle/fluid/distributed/collective/gather.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

void gather(GatherCommOptions* opts) {
  const auto& context = opts->context;
  gloo::transport::UnboundBuffer* in = opts->in.get();
  gloo::transport::UnboundBuffer* out = opts->out.get();
  const auto slot = gloo::Slot::build(gloo::kGatherSlotPrefix, opts->tag);

  PADDLE_ENFORCE(opts->root >= 0 && opts->root < context->size);
  PADDLE_ENFORCE(in);
  PADDLE_ENFORCE(opts->elementSize > 0);

  if (context->rank == opts->root) {
    PADDLE_ENFORCE(out);
    PADDLE_ENFORCE(out->size >= in->size * context->size);
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
