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

#include "paddle/fluid/operators/distributed/variable_response.h"

#include <string>
#include <utility>
#include <vector>
#ifdef PADDLE_WITH_CUDA
#include <nccl.h>
#endif
#include "paddle/fluid/platform/profiler.h"

#include "paddle/fluid/operators/distributed/send_recv.pb.h"
#include "paddle/fluid/operators/distributed/sendrecvop_utils.h"

namespace paddle {
namespace operators {
namespace distributed {

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

bool ReadRaw(::google::protobuf::io::CodedInputStream* input,
             const platform::DeviceContext& dev_ctx, platform::Place place,
             void* dest, int size) {
  const void* data = NULL;
  int size_to_write = 0;
  int length = size;
  int total_written = 0;

  if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
    auto& gpu_dev_ctx =
        static_cast<const platform::CUDADeviceContext&>(dev_ctx);
    platform::CPUPlace cpu;

    char* p = reinterpret_cast<char*>(dest);
    while (total_written < length) {
      if (!input->GetDirectBufferPointer(&data, &size_to_write)) {
        return false;
      }
      // NOTE: if raw buffer is large and have two neighbor fields of raw
      // buffers GetDirectBufferPointer can get all of them, use length to
      // truncate it.
      if (total_written + size_to_write > length) {
        size_to_write = length - total_written;
      }
      memory::Copy(boost::get<platform::CUDAPlace>(place),
                   reinterpret_cast<void*>(p), cpu, data, size_to_write,
                   gpu_dev_ctx.stream());
      p += size_to_write;
      total_written += size_to_write;

      input->Skip(size_to_write);
    }
    gpu_dev_ctx.Wait();
#else
    PADDLE_THROW("Unexpected branch");
#endif
    return true;
  }

  char* p = reinterpret_cast<char*>(dest);
  while (total_written < length) {
    if (!input->GetDirectBufferPointer(&data, &size_to_write)) {
      return false;
    }
    // NOTE: if raw buffer is large and have two neighbor fields of raw buffers
    // GetDirectBufferPointer can get all of them, use length to truncate it.
    if (total_written + size_to_write > length) {
      size_to_write = length - total_written;
    }
    // TODO(gongwb): can we avoid copy?
    platform::CPUPlace cpu;
    memory::Copy(cpu, reinterpret_cast<void*>(p), cpu, data, size_to_write);

    p += size_to_write;
    total_written += size_to_write;

    input->Skip(size_to_write);
  }

  return true;
}

bool VariableResponse::CopyLodTensorData(
    ::google::protobuf::io::CodedInputStream* input,
    const platform::DeviceContext& ctx, const framework::DDim& dims,
    int length) {
  auto* tensor = GetVar()->GetMutable<framework::LoDTensor>();
  tensor->Resize(dims);

  framework::LoD lod;
  for (int i = 0; i < meta_.lod_level(); ++i) {
    framework::Vector<size_t> v;
    for (int j = 0; j < meta_.lod(i).lod_data_size(); ++j) {
      v.push_back(meta_.lod(i).lod_data(j));
    }
    lod.push_back(v);
  }
  tensor->set_lod(lod);

  void* tensor_data =
      tensor->mutable_data(ctx.GetPlace(), ToTypeIndex(meta_.data_type()));

  if (!ReadRaw(input, ctx, tensor->place(), tensor_data, length)) {
    return false;
  }

  return true;
}

inline framework::DDim GetDims(
    const ::google::protobuf::RepeatedField<::google::protobuf::int64>& dims) {
  std::vector<int> vecdims;
  for (auto& d : dims) {
    vecdims.push_back(d);
  }
  return framework::make_ddim(vecdims);
}

bool VariableResponse::CopySelectRowsTensorData(
    ::google::protobuf::io::CodedInputStream* input,
    const platform::DeviceContext& ctx, const framework::DDim& dims,
    int length) {
  auto* slr = GetVar()->GetMutable<framework::SelectedRows>();
  slr->set_height(meta_.slr_height());
  auto* tensor = slr->mutable_value();
  tensor->Resize(dims);
  PADDLE_ENFORCE_EQ(static_cast<size_t>(tensor->numel()),
                    length / framework::SizeOfType(
                                 paddle::operators::distributed::ToTypeIndex(
                                     meta_.data_type())));
  void* tensor_data = tensor->mutable_data(
      ctx.GetPlace(),
      paddle::operators::distributed::ToTypeIndex(meta_.data_type()));

  if (!ReadRaw(input, ctx, tensor->place(), tensor_data, length)) {
    return false;
  }

  return true;
}

bool VariableResponse::CopySelectRowsData(
    ::google::protobuf::io::CodedInputStream* input,
    const platform::DeviceContext& ctx, int length) {
  auto* slr = GetVar()->GetMutable<framework::SelectedRows>();
  slr->mutable_rows()->resize(length /
                              framework::SizeOfType(typeid(int64_t)));  // int64
  int64_t* rows_data = slr->mutable_rows()->data();

  // copy rows CPU data, GPU data will be copied lazily.
  platform::CPUPlace cpu;
  if (!ReadRaw(input, ctx, cpu, rows_data, length)) {
    return false;
  }

  return true;
}

bool ParseLodData(::google::protobuf::io::CodedInputStream* input,
                  std::vector<int64_t>* lod) {
  while (true) {
    auto p = input->ReadTagWithCutoff(127);
    int tag = GetTagFieldNumber(p.first);
    WireType wt = GetTagWireType(p.first);

    if (!p.second) {
      return (tag == 0);
    }

    switch (tag) {
      case sendrecv::VariableMessage_LodData::kLodDataFieldNumber: {
        uint64_t v;
        if (wt == WIRETYPE_VARINT) {
          if (!input->ReadVarint64(&v)) {
            return false;
          }
          lod->push_back(v);
          break;
        }

        if (wt == WIRETYPE_LENGTH_DELIMITED) {
          int num_bytes = 0;
          if (!input->ReadVarintSizeAsInt(&num_bytes)) {
            return tag;
          }
          int start_pos = input->CurrentPosition();
          while (input->CurrentPosition() - start_pos < num_bytes) {
            uint64_t v;
            if (!input->ReadVarint64(&v)) {
              return tag;
            }
            lod->push_back(v);
          }
          break;
        }

        return false;
      }
      default: { return false; }
    }
  }

  return true;
}

int VariableResponse::Parse(const ::grpc::ByteBuffer& byte_buffer) {
  GrpcByteBufferSource source;
  source.Init(byte_buffer);
  GrpcByteBufferSourceWrapper r(&source);

  return Parse(&r);
}

int VariableResponse::Parse(Source* source) {
  ::google::protobuf::io::ZeroCopyInputStream* input_stream =
      source->contents();
  ::google::protobuf::io::CodedInputStream input(input_stream);
  input.SetTotalBytesLimit(INT_MAX, INT_MAX);

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
        break;
      }
      case sendrecv::VariableMessage::kTypeFieldNumber: {
        uint32_t v;
        if ((wt != WIRETYPE_VARINT) || !input.ReadVarint32(&v)) {
          return tag;
        }

        meta_.set_type(static_cast<::sendrecv::VarType>(v));
        break;
      }
      case sendrecv::VariableMessage::kDataTypeFieldNumber: {
        uint32_t v = 0;
        if ((wt != WIRETYPE_VARINT) || !input.ReadVarint32(&v)) {
          return tag;
        }

        meta_.set_data_type(static_cast<::sendrecv::VariableMessage_Type>(v));
        break;
      }
      case sendrecv::VariableMessage::kDimsFieldNumber: {
        // not packed
        if (wt == WIRETYPE_VARINT) {
          uint64_t v;
          if (!input.ReadVarint64(&v)) {
            return tag;
          }
          meta_.add_dims(v);
          break;
        }

        // packed
        if (wt == WIRETYPE_LENGTH_DELIMITED) {
          int num_bytes = 0;
          if (!input.ReadVarintSizeAsInt(&num_bytes)) {
            return tag;
          }
          int start_pos = input.CurrentPosition();
          while (input.CurrentPosition() - start_pos < num_bytes) {
            uint64_t v;
            if (!input.ReadVarint64(&v)) {
              return tag;
            }
            meta_.add_dims(v);
          }
          break;
        }
        return tag;
      }
      case sendrecv::VariableMessage::kLodLevelFieldNumber: {
        uint64_t v = 0;
        if ((wt != WIRETYPE_VARINT) || !input.ReadVarint64(&v)) {
          return tag;
        }
        meta_.set_lod_level(static_cast<int64_t>(v));
        break;
      }
      case sendrecv::VariableMessage::kLodFieldNumber: {
        int length = 0;
        if (wt != WIRETYPE_LENGTH_DELIMITED ||
            !ReadVarintSizeAsInt(&input, &length)) {
          return tag;
        }

        std::pair<::google::protobuf::io::CodedInputStream::Limit, int> p =
            input.IncrementRecursionDepthAndPushLimit(length);

        std::vector<int64_t> lod_data;
        if (p.second < 0 || !ParseLodData(&input, &lod_data)) {
          return tag;
        }

        if (!input.DecrementRecursionDepthAndPopLimit(p.first)) {
          return false;
        }

        if (lod_data.size() == 0) {
          break;
        }

        auto lod = meta_.add_lod();
        for (uint32_t i = 0; i < lod_data.size(); i++) {
          lod->add_lod_data(lod_data[i]);
        }
        break;
      }
      case sendrecv::VariableMessage::kSlrHeightFieldNumber: {
        uint64_t v = 0;
        if ((wt != WIRETYPE_VARINT) || !input.ReadVarint64(&v)) {
          return tag;
        }
        meta_.set_slr_height(static_cast<int64_t>(v));
        break;
      }
      case sendrecv::VariableMessage::kSerializedFieldNumber: {
        PADDLE_ENFORCE((meta_.type() == sendrecv::SELECTED_ROWS ||
                        meta_.type() == sendrecv::LOD_TENSOR ||
                        meta_.type() == sendrecv::NCCL_ID) &&
                           meta_.varname() != "",
                       "meta info should be got first!");

        int num_bytes = 0;
        if (wt != WIRETYPE_LENGTH_DELIMITED ||
            !ReadVarintSizeAsInt(&input, &num_bytes)) {
          return tag;
        }

        if (meta_.type() == sendrecv::NCCL_ID) {
#ifdef PADDLE_WITH_CUDA
          auto* var = scope_->FindVar(meta_.varname());
          if (var != nullptr) {
            ncclUniqueId* id = var->GetMutable<ncclUniqueId>();
            if (!ReadRaw(&input, *dev_ctx_, platform::CPUPlace(), id->internal,
                         num_bytes)) {
              return tag;
            }
          }
          break;
#else
          PADDLE_THROW("Not compiled with CUDA!");
#endif
        }

        framework::DDim dims = GetDims(meta_.dims());
        if (meta_.type() == sendrecv::LOD_TENSOR) {
          PADDLE_ENFORCE(meta_.lod_size() >= 0,
                         "lod info should be got first!");
          if (!CopyLodTensorData(&input, *dev_ctx_, dims, num_bytes)) {
            return tag;
          }
          break;
        }

        if (meta_.type() == sendrecv::SELECTED_ROWS) {
          if (!CopySelectRowsTensorData(&input, *dev_ctx_, dims, num_bytes)) {
            return tag;
          }
          break;
        }

        return tag;
      }
      case sendrecv::VariableMessage::kRowsFieldNumber: {
        PADDLE_ENFORCE((meta_.type() == sendrecv::SELECTED_ROWS ||
                        meta_.type() == sendrecv::LOD_TENSOR) &&
                           meta_.varname() != "",
                       "meta info should be got first!");

        int num_bytes = 0;
        if (wt != WIRETYPE_LENGTH_DELIMITED ||
            !ReadVarintSizeAsInt(&input, &num_bytes)) {
          return tag;
        }

        if (!CopySelectRowsData(&input, *dev_ctx_, num_bytes)) {
          return tag;
        }
        break;
      }
      case sendrecv::VariableMessage::kOutVarnameFieldNumber: {
        uint32_t length;
        if ((wt != WIRETYPE_LENGTH_DELIMITED) || !input.ReadVarint32(&length)) {
          return tag;
        }

        std::string temp;
        if (!input.ReadString(&temp, length)) {
          return tag;
        }

        meta_.set_out_varname(temp);
        break;
      }
      case sendrecv::VariableMessage::kProfileFieldNumber: {
        uint64_t profiling = 0;
        if (!input.ReadVarint64(&profiling)) {
          return tag;
        }
        meta_.set_profile(profiling);
        int64_t listener_id = platform::ListenerId();
        if (listener_id <= 0) {
          break;
        }
        if (profiling == platform::kEnableProfiler &&
            !platform::IsProfileEnabled()) {
          platform::EnableProfiler(platform::ProfilerState::kCPU);
        } else if (profiling == platform::kDisableProfiler &&
                   platform::IsProfileEnabled()) {
          // TODO(panyx0718): Should we allow to customize file dir.
          platform::DisableProfiler(
              platform::EventSortingKey::kDefault,
              string::Sprintf("/tmp/profile_ps_%lld", listener_id));
        }
        break;
      }
      default: {
        // Unknown tag, return unknown error.
        return -1;
      }
    }
  }

  return 0;
}

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
