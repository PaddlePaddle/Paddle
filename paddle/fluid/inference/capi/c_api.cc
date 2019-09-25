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

#include "paddle/fluid/inference/capi/c_api.h"
#include <algorithm>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/inference/capi/c_api_internal.h"

extern "C" {

PD_PaddleBuf* PD_NewPaddleBuf() { return new PD_PaddleBuf; }

void PD_DeletePaddleBuf(PD_PaddleBuf* buf) {
  if (buf) {
    delete buf;
    buf = nullptr;
  }
}

void PD_PaddleBufResize(PD_PaddleBuf* buf, size_t length) {
  buf->buf.Resize(length);
}

void PD_PaddleBufReset(PD_PaddleBuf* buf, void* data, size_t length) {
  buf->buf.Reset(data, length);
}

bool PD_PaddleBufEmpty(PD_PaddleBuf* buf) { return buf->buf.empty(); }

void* PD_PaddleBufData(PD_PaddleBuf* buf) { return buf->buf.data(); }

size_t PD_PaddleBufLength(PD_PaddleBuf* buf) { return buf->buf.length(); }

void PD_PaddleBufAssign(PD_PaddleBuf* buf_des, PD_PaddleBuf* buf_ori) {
  buf_des->buf = buf_ori->buf;
}
}  // extern "C"
