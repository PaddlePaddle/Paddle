//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "tensor_parser.h"
#include <string.h>
#include "paddle/fluid/operators/detail/send_recv.pb.h"

namespace paddle {
namespace operators {
namespace detail {

enum WireType {
  WIRETYPE_VARINT = 0,
  WIRETYPE_LENGTH_DELIMITED = 2,
};

inline int GetTagFieldNumber(uint32_t tag) { return tag >> 3; }

inline WireType GetTagWireType(uint32_t tag) {
  return static_cast<WireType>(tag & 0x7);
}

bool ReadVarintSizeAsInt(::google::protobuf::io::CodedInputStream* input,
                         int* result) {
  uint64_t v;
  if (input->ReadVarint64(&v) && v <= static_cast<uint64_t>(INT_MAX)) {
    *result = static_cast<int>(v);
    return true;
  } else {
    return false;
  }
}

/*
bool TensorResponse::ParseTensorSubmessage(
    protobuf::io::CodedInputStream* input, TensorProto* tensor_meta) {
  bool seen_tensor_content = false;
  while (true) {
    auto p = input->ReadTagWithCutoff(127);
    int tag     = GetTagFieldNumber(p.first);
    WireType wt = GetTagWireType(p.first);

    if (!p.second) {
      bool ok = (tag == 0);
      if (ok && !seen_tensor_content) {
        // No tensor content: could be because it's a zero-length tensor
        tensor_ = std::move(t);
      }
      return ok;
    }

    switch (tag) {
      case TensorProto::kDtypeFieldNumber: {
        uint32 v;
        if ((wt != WIRETYPE_VARINT) || !input->ReadVarint32(&v))
            return false;

        if (seen_tensor_content) return false;
        tensor_meta->set_dtype(static_cast<DataType>(static_cast<int>(v)));
        if (!DataTypeCanUseMemcpy(tensor_meta->dtype())) return false;
        break;
      }
      case TensorProto::kTensorShapeFieldNumber: {
        if ((wt != WIRETYPE_LENGTH_DELIMITED) ||
            !ReadNestedMessage(input, tensor_meta->mutable_tensor_shape()))
          return false;
        if (seen_tensor_content) return false;
        break;
      }
      case TensorProto::kVersionNumberFieldNumber: {
        uint32 v;
        if ((wt != WIRETYPE_VARINT) || !input->ReadVarint32(&v)) return false;
        if (seen_tensor_content) return false;
        tensor_meta->set_version_number(static_cast<int32>(v));
        break;
      }
      case TensorProto::kTensorContentFieldNumber: {
        // If we haven't seen the dtype and tensor_shape data first, we can't
        // deal with this in the fast path.
        if (seen_tensor_content) return false;
        if (wt != WIRETYPE_LENGTH_DELIMITED ||
            !tensor_meta->has_tensor_shape()) {
          return false;
        }
        int num_bytes;
        if (!ReadVarintSizeAsInt(input, &num_bytes)) return false;
        seen_tensor_content = true;
        TensorShape shape(tensor_meta->tensor_shape());
        Tensor t(allocator_, tensor_meta->dtype(), shape);
        StringPiece buf = t.tensor_data();
        if (static_cast<size_t>(num_bytes) != buf.size()) return false;
        // TODO(jeff,sanjay): Figure out a way to avoid this copy if
        // the underlying ZeroCopyInputStream data is properly aligned
        // and compatible with what allocator_ wants.
        if (!input->ReadRaw(const_cast<char*>(buf.data()), num_bytes))
          return false;
        tensor_ = std::move(t);
        break;
      }
      default: {
        // Some other tag our fast path code is not prepared to handle.
        // return false.
        return false;
      }
    }
  }
}
*/

bool ReadRaw(::google::protobuf::io::CodedInputStream* input,
             const platform::DeviceContext& dev_ctx, platform::Place place,
             void* buf, int size) {
  const void* data = NULL;
  int size_to_write = 0;

  if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
    auto& gpu_dev_ctx =
        static_cast<const platform::CUDADeviceContext&>(dev_ctx);
    platform::CPUPlace cpu;

    char* p = reinterpret_cast<char*>(buf);
    while (size != 0) {
      if (input->GetDirectBufferPointer(&data, &size_to_write)) {
        return false;
      }

      memory::Copy(boost::get<platform::CUDAPlace>(place),
                   reinterpret_cast<void*>(p), cpu, data, size_to_write,
                   gpu_dev_ctx.stream());

      p += size_to_write;
      size -= size_to_write;

      input->Skip(size_to_write);
    }
    gpu_dev_ctx.Wait();
#else
    PADDLE_THROW("Unexpected branch");
#endif
    return true;
  }

  char* p = reinterpret_cast<char*>(buf);
  while (size != 0) {
    if (input->GetDirectBufferPointer(&data, &size_to_write)) {
      return false;
    }
    // TODO(gongwb): don't copy if it's aligned?
    memcpy(reinterpret_cast<void*>(p), data, size_to_write);

    p += size_to_write;
    size -= size_to_write;

    input->Skip(size_to_write);
  }

  return true;
}

bool CopyLodTensorData(::google::protobuf::io::CodedInputStream* input) {
  /*
  if (ReadRaw(&input, tensor->place(),
              reinterpret_cast<const void*>tensor->data(), length)){
      return false;
  }
  */

  return true;
}

bool CopySelectRowsTensorData(::google::protobuf::io::CodedInputStream* input) {
  return true;
}

bool CopySelectRowsData(::google::protobuf::io::CodedInputStream* input) {
  return true;
}

int TensorResponse::Parse(::grpc::ByteBuffer& byte_buffer,
                          const platform::DeviceContext& dev_ctx) {
  GrpcByteBufferSource source;
  source.Init(byte_buffer);

  ::google::protobuf::io::ZeroCopyInputStream* input_stream = &source;
  ::google::protobuf::io::CodedInputStream input(input_stream);
  input.SetTotalBytesLimit(INT_MAX, INT_MAX);

  // framework::Variable* var = NULL;
  // meta_.set_type(-1);
  int tensor_type = -1;

  while (true) {
    auto p = input.ReadTagWithCutoff(127);
    int tag = GetTagFieldNumber(p.first);
    WireType wt = GetTagWireType(p.first);
    if (!p.second) {
      if (tag != 0) {
        return -1;
      }

      return 0;
    }

    switch (tag) {
      case sendrecv::VariableMessage::kVarnameFieldNumber: {
        uint32_t length;
        if ((wt != WIRETYPE_LENGTH_DELIMITED) || !input.ReadVarint32(&length)) {
          return tag;
        }

        std::string temp;
        if (!input.ReadString(&temp, length)) {
          return tag;
        }

        meta_.set_varname(temp);
        // var = scope_->FindVar(temp);
        break;
      }
      case sendrecv::VariableMessage::kTypeFieldNumber: {
        uint64_t v;
        if ((wt != WIRETYPE_VARINT) || !input.ReadVarint64(&v)) {
          return tag;
        }

        tensor_type = static_cast<int64_t>(v);
        meta_.set_type(static_cast<::sendrecv::VarType>(v));
        break;
      }
      case sendrecv::VariableMessage::kDataTypeFieldNumber: {
        uint64_t v;
        if ((wt != WIRETYPE_VARINT) || !input.ReadVarint64(&v)) {
          return tag;
        }

        meta_.set_data_type(static_cast<::sendrecv::VariableMessage_Type>(v));
        break;
      }
      case sendrecv::VariableMessage::kDimsFieldNumber: {
        uint64_t v;
        if ((wt != WIRETYPE_LENGTH_DELIMITED) || !input.ReadVarint64(&v)) {
          return tag;
        }

        meta_.add_dims(v);
        break;
      }
      case sendrecv::VariableMessage::kLodLevelFieldNumber: {
        uint64_t v;
        if ((wt != WIRETYPE_VARINT) || !input.ReadVarint64(&v)) {
          return tag;
        }
        meta_.set_lod_level(static_cast<int64_t>(v));
        break;
      }
      case sendrecv::VariableMessage::kLodFieldNumber: {
        int length;
        if (wt != WIRETYPE_LENGTH_DELIMITED ||
            !ReadVarintSizeAsInt(&input, &length)) {
          return tag;
        }
        break;
      }
      case sendrecv::VariableMessage::kSerializedFieldNumber: {
        PADDLE_ENFORCE(tensor_type > 0 &&
                           (meta_.type() == sendrecv::LOD_TENSOR ||
                            meta_.type() == sendrecv::SELECTED_ROWS) &&
                           meta_.varname() != "",
                       "tensor_type and varname should be got first!");

        int length;
        if (wt != WIRETYPE_LENGTH_DELIMITED ||
            !ReadVarintSizeAsInt(&input, &length)) {
          return tag;
        }

        if (meta_.type() == sendrecv::LOD_TENSOR) {
          if (!CopyLodTensorData(&input)) {
            return tag;
          }
          break;
        }

        if (meta_.type() == sendrecv::LOD_TENSOR) {
          if (!CopySelectRowsData(&input)) {
            return tag;
          }
          break;
        }

        break;
      }
      case sendrecv::VariableMessage::kRowsFieldNumber: {
        PADDLE_ENFORCE(
            meta_.type() == sendrecv::SELECTED_ROWS && meta_.varname() != "",
            "tensor_type and varname should be got first!");

        int length = 0;
        if (wt != WIRETYPE_LENGTH_DELIMITED ||
            !ReadVarintSizeAsInt(&input, &length)) {
          return tag;
        }

        if (CopySelectRowsTensorData(&input)) {
          return tag;
        }
        break;
      }

      default: {
        // Unknown tag, so don't handle we can't handle on the fast path
        return -1;
      }
    }
  }

  return 0;
}

};  // namespace detail
};  // namespace operators
};  // namespace paddle
