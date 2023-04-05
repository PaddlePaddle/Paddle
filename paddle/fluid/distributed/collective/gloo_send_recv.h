/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "gloo/context.h"
#include "gloo/transport/unbound_buffer.h"

namespace paddle {
namespace distributed {

constexpr uint8_t kSendRecvSlotPrefix = 0x08;

class SendRecvOptions {
 public:
  explicit SendRecvOptions(const std::shared_ptr<gloo::Context>& context)
      : context(context), timeout(context->getTimeout()) {}

  template <typename T>
  void setInput(T* ptr, size_t elements) {
    this->in = context->createUnboundBuffer(ptr, elements * sizeof(T));
  }

  template <typename T>
  void setOutput(T* ptr, size_t elements) {
    this->out = context->createUnboundBuffer(ptr, elements * sizeof(T));
  }

  void setSrc(int src) { this->src = src; }

  void setDst(int dst) { this->dst = dst; }

  void setTag(uint32_t tag) { this->tag = tag; }

  void setTimeout(std::chrono::milliseconds timeout) {
    this->timeout = timeout;
  }

 protected:
  std::shared_ptr<gloo::Context> context;
  std::unique_ptr<gloo::transport::UnboundBuffer> in;
  std::unique_ptr<gloo::transport::UnboundBuffer> out;

  // Rank of process to send_recv from.
  int src = -1;

  // Rank of process to send_recv to.
  int dst = -1;

  // Tag for this operation.
  // Must be unique across operations executing in parallel.
  uint32_t tag = 0;

  // End-to-end timeout for this operation.
  std::chrono::milliseconds timeout;

  friend void send_recv(SendRecvOptions*);
};

void send_recv(SendRecvOptions* opts);

}  // namespace distributed
}  // namespace paddle
