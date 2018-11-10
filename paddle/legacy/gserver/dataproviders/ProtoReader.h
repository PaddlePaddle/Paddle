/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <memory>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message_lite.h>

namespace paddle {

/**
 * ProtoReader/ProtoWriter are used to read/write a sequence of protobuf
 * messages from/to i/ostream.
 */
class ProtoReader {
 public:
  explicit ProtoReader(std::istream* s, bool dataCompression = false) {
    CHECK(s) << "istream pointer is nullptr";
    istreamInput_.reset(new google::protobuf::io::IstreamInputStream(s));
    if (dataCompression) {
      gzipInput_.reset(
          new google::protobuf::io::GzipInputStream(istreamInput_.get()));
      codedInput_.reset(
          new google::protobuf::io::CodedInputStream(gzipInput_.get()));
    } else {
      codedInput_.reset(
          new google::protobuf::io::CodedInputStream(istreamInput_.get()));
    }
    dataCompression_ = dataCompression;
    approximateReadedBytes_ = 0;
    codedInput_->SetTotalBytesLimit(kDefaultTotalBytesLimit,
                                    kDefaultTotalBytesLimit);
  }

  /**
   * read one message
   */
  bool read(google::protobuf::MessageLite* msg) {
    if (approximateReadedBytes_ >= kMaxLimitBytes) {
      // Once bytes we read get close to 64MB(larger than 55MB),
      // we re-intialize the codedInputStream object.
      approximateReadedBytes_ = 0;

      /**
       * Explicitly destroys the object owned by unique_ptr at first and then
       * construct an new object.
       *
       * 1.reset()
       *
       * 2.reset(new ...)   <-- such sequence is EXTREAMLY important!
       *
       * Reason: (!!!Read me before you modify the following 2 lines of
       * codes!!!)
       *
       * Otherwise, reset() method will ask the CodedInputStream constructor
       * to construct the new object at first forcing the IstreamInputStream
       * object to move its underlying pointer to the next 8192 bytes.
       *
       * Then the old object will be destroied calling
       * IstreamInputStream::BackUp() to move the underlying pointer back.
       * This means that the InstreamInputStream object is referenced by
       * 2 different CodedInputStream object at the same time which "confuses"
       * the position of istreamInput_'s underlying pointer. Such fatal
       * confusion will lead to undefined behaviour when 'codedInput_' is
       * used to read new data.
       *
       */
      codedInput_.reset();
      if (dataCompression_) {
        codedInput_.reset(
            new google::protobuf::io::CodedInputStream(gzipInput_.get()));
      } else {
        codedInput_.reset(
            new google::protobuf::io::CodedInputStream(istreamInput_.get()));
      }
      codedInput_->SetTotalBytesLimit(kDefaultTotalBytesLimit,
                                      kDefaultTotalBytesLimit);
    }

    uint32_t size;
    if (!codedInput_->ReadVarint32(&size)) {
      return false;
    }
    google::protobuf::io::CodedInputStream::Limit limit =
        codedInput_->PushLimit(size);
    CHECK(msg->ParseFromCodedStream(codedInput_.get()));
    codedInput_->PopLimit(limit);

    /**
     * size is varint in the data file, we don't know the length.
     * We assume every size takes 4 bytes in the data file.
     */
    approximateReadedBytes_ += 4 + size;
    return true;
  }

 protected:
  std::unique_ptr<google::protobuf::io::ZeroCopyInputStream> istreamInput_;
  std::unique_ptr<google::protobuf::io::GzipInputStream> gzipInput_;
  std::unique_ptr<google::protobuf::io::CodedInputStream> codedInput_;
  bool dataCompression_;

  /**
   * This is the maximum number of bytes that this CodedInputStream will read
   * before refusing to continue.
   */
  static const int kDefaultTotalBytesLimit = 64 << 20;  // 64MB

  /**
   * If data readed by the reader is more than 55MB( << 64MB),
   * we reset the CodedInputStream object.
   * This can help avoid 64MB warning which will cause the ParseFromCodedStream
   * to fail.
   */
  static const int kMaxLimitBytes = 55 << 20;

  /**
   * This variable dosen't store the exact bytes readed by CodedInputStream
   * object since which is constructed. Instead, it store the approximate bytes
   * because we can't tell how many bytes are readed by the object with the
   * help of API.
   *
   * @note this code depends on protobuf 2.4.0. There is nothing like
   * CodedInputStream::CurrentPosition() in protobuf 2.5.0 to tell us how many
   * bytes has the object readed so far. Therefore, we calculated bytes
   * ourselves.
   */
  int approximateReadedBytes_;
};

class ProtoWriter {
 public:
  explicit ProtoWriter(std::ostream* s, bool dataCompression = false) {
    CHECK(s) << "ostream pointer is nullptr";
    ostreamOutput_.reset(new google::protobuf::io::OstreamOutputStream(s));
    if (dataCompression) {
      gzipOutput_.reset(
          new google::protobuf::io::GzipOutputStream(ostreamOutput_.get()));
      codedOutput_.reset(
          new google::protobuf::io::CodedOutputStream(gzipOutput_.get()));
    } else {
      codedOutput_.reset(
          new google::protobuf::io::CodedOutputStream(ostreamOutput_.get()));
    }
  }

  /**
   * write one message.
   */
  bool write(const google::protobuf::MessageLite& msg) {
    codedOutput_->WriteVarint32(msg.ByteSize());
    bool ret = msg.SerializeToCodedStream(codedOutput_.get());
    return ret;
  }

 protected:
  std::unique_ptr<google::protobuf::io::ZeroCopyOutputStream> ostreamOutput_;
  std::unique_ptr<google::protobuf::io::GzipOutputStream> gzipOutput_;
  std::unique_ptr<google::protobuf::io::CodedOutputStream> codedOutput_;
};

}  // namespace paddle
