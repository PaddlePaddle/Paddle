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

#include "paddle/fluid/operators/reader/pipe_reader.h"
#include <algorithm>

namespace paddle {
namespace operators {
namespace reader {

struct DeserializedDataFunctor {
  DeserializedDataFunctor(void** buf, framework::Tensor* tensor,
                          const platform::Place& place)
      : buf_(buf), tensor_(tensor), place_(place) {}

  template <typename T>
  void apply() {
    *buf_ = tensor_->mutable_data<T>(place_);
  }

  void** buf_;
  framework::Tensor* tensor_;
  platform::Place place_;
};

PipeReader::PipeReader(const int pipe_fd, size_t capacity)
    : framework::FileReader() {
  pipe_ = std::unique_ptr<ReadPipe>(new ReadPipe(pipe_fd));
  queue_ = std::unique_ptr<LoDTensorBlockingQueue>(
      new LoDTensorBlockingQueue(capacity));
}

PipeReader::~PipeReader() {}

void PipeReader::Start() {
  thread_.reset(new std::thread(&PipeReader::ThreadFunc, this));
}

void PipeReader::Shutdown() { thread_->join(); }

void PipeReader::ReadNext(std::vector<framework::LoDTensor>* out) {
  bool success;
  *out = queue_->Pop(&success);
  if (!success) out->clear();
}

void PipeReader::ThreadFunc() {
  queue_->ReOpen();
  try {
    while (true) {
      // Pipe data structure
      //
      // num_tensor: (uint64) number of output tensors
      // num_tensor x {
      //      lod_level: (uint64) level of lod
      //      lod_level x {
      //          level_size: (uint64) size of each lod level
      //          lod_data: (level_size * uint64) data of each lod level
      //      }
      //      dtype: (uint32) enum dtype
      //      num_dim: (uint64) number of dims
      //      num_dim x {
      //          dim_data: (int64): dim size
      //      }
      //      tensor_data: (prod(dims) * sizeof(dtype)) data of tensor
      // }
      std::vector<framework::LoDTensor> out;
      uint64_t num_tensor;
      pipe_->read(reinterpret_cast<uint8_t*>(&num_tensor), sizeof(num_tensor));
      if (num_tensor == 0) {
        break;
      }
      out.resize(num_tensor);
      for (uint64_t i = 0; i < num_tensor; i++) {
        auto& tensor = out[i];
        auto& lod = *tensor.mutable_lod();
        // #levels
        uint64_t lod_level;
        pipe_->read(reinterpret_cast<uint8_t*>(&lod_level), sizeof(lod_level));
        lod.resize(lod_level);
        for (uint64_t j = 0; j < lod_level; j++) {
          // level size
          uint64_t level_size;
          pipe_->read(reinterpret_cast<uint8_t*>(&level_size),
                      sizeof(level_size));
          // level data
          std::vector<uint64_t> tmp(level_size);
          pipe_->read(reinterpret_cast<uint8_t*>(tmp.data()),
                      level_size * sizeof(uint64_t));
          lod[j] = std::vector<size_t>(tmp.begin(), tmp.end());
        }
        // dtype
        uint32_t dtype_enum;
        pipe_->read(reinterpret_cast<uint8_t*>(&dtype_enum),
                    sizeof(dtype_enum));
        framework::proto::VarType::Type dtype =
            static_cast<framework::proto::VarType::Type>(dtype_enum);
        auto ctx = platform::CPUDeviceContext();
        tensor.mutable_data(ctx.GetPlace(), dtype);
        // num_dims
        uint64_t num_dim;
        pipe_->read(reinterpret_cast<uint8_t*>(&num_dim), sizeof(num_dim));
        std::vector<int64_t> dims;
        dims.resize(num_dim);
        pipe_->read(reinterpret_cast<uint8_t*>(dims.data()),
                    sizeof(uint64_t) * num_dim);
        tensor.Resize(framework::make_ddim(dims));
        size_t size = tensor.numel() * framework::SizeOfType(dtype);
        void* buf;
        framework::VisitDataType(
            dtype, DeserializedDataFunctor(&buf, &tensor, ctx.GetPlace()));
        pipe_->read(reinterpret_cast<uint8_t*>(buf), size);
      }
      queue_->Push(out);
    }
  } catch (...) {
  }
  queue_->Close();
}

}  // namespace reader
}  // namespace operators
}  // namespace paddle
