/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/utils/Util.h"

#include <sys/uio.h>

#include <memory>
#include <vector>

struct sxi_sock;

namespace paddle {

class SocketChannel;
enum ChannelType {
  F_TCP = 1,
  F_RDMA = 2,
};

/// reading a set of blocks of data from SocketChannel.
class MsgReader {
public:
  MsgReader(SocketChannel* channel, size_t numIovs);
  ~MsgReader() {
    /// ensure all data blocks have been processed
    CHECK_EQ(currentBlockIndex_, blockLengths_.size());
  }
  /**
   * @brief number of remaining parts
   */
  size_t getNumBlocks() const {
    return blockLengths_.size() - currentBlockIndex_;
  }

  /**
   * @brief lenght of next block
   */
  size_t getNextBlockLength() const { return getBlockLength(0); }

  /**
   * @brief get the total length of all the remaining blocks
   */
  size_t getTotalLength() const {
    size_t total = 0;
    for (size_t i = currentBlockIndex_; i < blockLengths_.size(); ++i) {
      total += blockLengths_[i];
    }
    return total;
  }

  /**
   * @brief Get the length for block currentBlockIndex + i
   */
  size_t getBlockLength(size_t i) const {
    return blockLengths_[currentBlockIndex_ + i];
  }

  /**
   * @brief  read blocks data and store it to buf
   */
  void readBlocks(const std::vector<void*>& bufs);
  void readNextBlock(void* buf);

protected:
  SocketChannel* channel_;
  std::vector<size_t> blockLengths_;
  size_t currentBlockIndex_;
};

/// APIs for reading and writing byte stream data or naive iov data
/// from the APIs both RDMA and TCP exhibits byte stream style
class SocketChannel {
public:
  SocketChannel(int socket, const std::string& peerName)
      : tcpSocket_(socket), peerName_(peerName) {
    tcpRdma_ = F_TCP;
  }
  SocketChannel(struct sxi_sock* socket, const std::string& peerName)
      : rdmaSocket_(socket), peerName_(peerName) {
    tcpRdma_ = F_RDMA;
  }

  ~SocketChannel();

  const std::string& getPeerName() const { return peerName_; }

  /**
   * @brief read size bytes.
   *
   * @note  keep reading until getting size bytes or sock is closed
   *        is closed
   */
  size_t read(void* buf, size_t size);

  /**
   * @brief write size bytes.
   *
   * @note  keep writing until writing size bytes or sock is closed
   */
  size_t write(const void* buf, size_t size);

  /**
   * @brief write a set of buffers.
   *
   * @note  keep writing until all buffers are written or sock is closed
   */
  size_t writev(const std::vector<struct iovec>& iov);

  /**
   * @brief read a set of buffers.
   *
   * @note  keep reading until all buffers are full or sock is closed.
   */
  size_t readv(std::vector<struct iovec>* iov);

  /**
   * @brief write a set of buffers.
   *
   * @note  keep writing until all buffers are passed or sock is closed
   */
  void writeMessage(const std::vector<struct iovec>& iov);

  /// return null to indicate socket is closed
  std::unique_ptr<MsgReader> readMessage();

protected:
  struct MessageHeader {
    int64_t totalLength;  /// include the header
    int64_t numIovs;
    int64_t iovLengths[0];
  };

  int tcpSocket_;
  struct sxi_sock* rdmaSocket_;
  const std::string peerName_;
  enum ChannelType tcpRdma_;
};

}  // namespace paddle
