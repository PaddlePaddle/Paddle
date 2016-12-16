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

#include "SocketChannel.h"

#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include "RDMANetwork.h"

#include "paddle/utils/Util.h"

namespace paddle {

/**
 * UIO_MAXIOV is documented in writev(2), but <sys/uio.h> only
 * declares it on osx/ios if defined(KERNEL)
 */
#ifndef UIO_MAXIOV
#define UIO_MAXIOV 512
#endif

SocketChannel::~SocketChannel() {
  if (tcpRdma_ == F_TCP)
    close(tcpSocket_);
  else
    rdma::close(rdmaSocket_);
  LOG(INFO) << "destory connection in socket channel, peer = " << peerName_;
}

size_t SocketChannel::read(void* buf, size_t size) {
  size_t total = 0;
  while (total < size) {
    ssize_t len;
    if (tcpRdma_ == F_TCP)
      len = ::read(tcpSocket_, (char*)buf + total, size - total);
    else
      len = rdma::read(rdmaSocket_, (char*)buf + total, size - total);

    PCHECK(len >= 0) << " peer=" << peerName_;
    if (len <= 0) {
      return total;
    }
    total += len;
  }
  return total;
}

size_t SocketChannel::write(const void* buf, size_t size) {
  size_t total = 0;
  while (total < size) {
    ssize_t len;
    if (tcpRdma_ == F_TCP)
      len = ::write(tcpSocket_, (const char*)buf + total, size - total);
    else
      len = rdma::write(rdmaSocket_, (char*)buf + total, size - total);

    PCHECK(len >= 0) << " peer=" << peerName_;
    if (len <= 0) {
      return total;
    }
    total += len;
  }
  return total;
}

template <class IOFunc, class SocketType>
static size_t readwritev(IOFunc iofunc,
                         SocketType socket,
                         iovec* iovs,
                         int iovcnt,
                         int maxiovs,
                         const std::string& peerName) {
  int curIov = 0;
  size_t total = 0;

  for (int i = 0; i < iovcnt; ++i) {
    total += iovs[i].iov_len;
  }

  size_t size = 0;
  size_t curIovSizeDone = 0;

  while (size < total) {
    ssize_t len =
        iofunc(socket, &iovs[curIov], std::min(iovcnt - curIov, maxiovs));
    PCHECK(len > 0) << " peer=" << peerName << " curIov=" << curIov
                    << " iovCnt=" << iovcnt
                    << " iovs[curIov].base=" << iovs[curIov].iov_base
                    << " iovs[curIov].iov_len=" << iovs[curIov].iov_len;
    size += len;

    /// restore iovs[curIov] to the original value
    iovs[curIov].iov_base =
        (void*)((char*)iovs[curIov].iov_base - curIovSizeDone);
    iovs[curIov].iov_len += curIovSizeDone;

    len += curIovSizeDone;

    while (curIov < iovcnt) {
      if ((size_t)len < iovs[curIov].iov_len) break;
      len -= iovs[curIov].iov_len;
      ++curIov;
    }
    if (curIov < iovcnt) {
      curIovSizeDone = len;
      iovs[curIov].iov_base = (void*)((char*)iovs[curIov].iov_base + len);
      iovs[curIov].iov_len -= len;
    }
  }
  return size;
}

/// rdma::readv and rdma::writev can take advantage of RDMA blocking offload
/// transfering
size_t SocketChannel::writev(const std::vector<struct iovec>& iovs) {
  if (tcpRdma_ == F_TCP)
    return readwritev(::writev,
                      tcpSocket_,
                      const_cast<iovec*>(&iovs[0]),
                      iovs.size(),
                      UIO_MAXIOV,
                      peerName_);
  else
    return readwritev(rdma::writev,
                      rdmaSocket_,
                      const_cast<iovec*>(&iovs[0]),
                      iovs.size(),
                      MAX_VEC_SIZE,
                      peerName_);
}

size_t SocketChannel::readv(std::vector<struct iovec>* iovs) {
  if (tcpRdma_ == F_TCP)
    return readwritev(::readv,
                      tcpSocket_,
                      const_cast<iovec*>(&(*iovs)[0]),
                      iovs->size(),
                      UIO_MAXIOV,
                      peerName_);
  else
    return readwritev(rdma::readv,
                      rdmaSocket_,
                      const_cast<iovec*>(&(*iovs)[0]),
                      iovs->size(),
                      MAX_VEC_SIZE,
                      peerName_);
}

void SocketChannel::writeMessage(const std::vector<struct iovec>& userIovs) {
  MessageHeader header;
  header.numIovs = userIovs.size();

  std::vector<size_t> iovLengths;
  iovLengths.reserve(userIovs.size());
  for (auto& iov : userIovs) {
    iovLengths.push_back(iov.iov_len);
  }

  std::vector<iovec> iovs;
  iovs.reserve(userIovs.size() + 2);
  iovs.push_back({&header, sizeof(header)});
  iovs.push_back({&iovLengths[0],
                  static_cast<size_t>(sizeof(iovLengths[0]) * header.numIovs)});
  iovs.insert(iovs.end(), userIovs.begin(), userIovs.end());

  header.totalLength = 0;
  for (auto& iov : iovs) {
    header.totalLength += iov.iov_len;
  }

  PCHECK(writev(iovs) == (size_t)header.totalLength);
}

std::unique_ptr<MsgReader> SocketChannel::readMessage() {
  MessageHeader header;

  size_t len = read(&header, sizeof(header));
  if (len == 0) {
    return nullptr;
  }

  PCHECK(len == sizeof(header));

  std::unique_ptr<MsgReader> msgReader(new MsgReader(this, header.numIovs));

  CHECK_EQ(msgReader->getTotalLength() + sizeof(header) +
               msgReader->getNumBlocks() * sizeof(size_t),
           (size_t)header.totalLength)
      << " totalLength=" << msgReader->getTotalLength()
      << " numBlocks=" << msgReader->getNumBlocks();
  return msgReader;
}

MsgReader::MsgReader(SocketChannel* channel, size_t numBlocks)
    : channel_(channel), blockLengths_(numBlocks), currentBlockIndex_(0) {
  size_t size = numBlocks * sizeof(blockLengths_[0]);
  PCHECK(channel_->read(&blockLengths_[0], size) == size);
}

void MsgReader::readBlocks(const std::vector<void*>& bufs) {
  CHECK_LE(currentBlockIndex_ + bufs.size(), blockLengths_.size());
  std::vector<iovec> iovs;
  iovs.reserve(bufs.size());
  size_t totalLength = 0;
  for (void* buf : bufs) {
    iovs.push_back({buf, getNextBlockLength()});
    totalLength += getNextBlockLength();
    ++currentBlockIndex_;
  }

  PCHECK(channel_->readv(&iovs) == totalLength);
}

void MsgReader::readNextBlock(void* buf) {
  CHECK_LT(currentBlockIndex_, blockLengths_.size());
  PCHECK(channel_->read(buf, getNextBlockLength()) == getNextBlockLength());
  ++currentBlockIndex_;
}

}  // namespace paddle
