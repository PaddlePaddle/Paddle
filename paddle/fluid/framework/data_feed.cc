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

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <utility>
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

#include "gflags/gflags.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/framework/data_feed.h"

DEFINE_bool(is_text_feed, false, "is_text_feed");

namespace paddle {
namespace framework {
void TextClassDataFeed::Init(const datafeed::DataFeedParameter& feed_param) {
  // hard coding for a specific datafeed
  feed_vec_.resize(2);
  // feed_vec_[0].reset(new LoDTensor);
  // feed_vec_[1].reset(new LoDTensor);
  all_slot_ids_ = {0, 1};
  use_slot_ids_ = {0, 1};
  use_slot_alias_ = {"words", "label"};

  file_content_buffer_host_.reset(new char[200*1024*1024],
                                  [](char *p) {delete[] p;});
  file_content_buffer_ = file_content_buffer_host_.get();
  file_content_buffer_ptr_ = file_content_buffer_;

  batch_id_host_.reset(new int[10240*1024],
                      [](int *p) {delete[] p;});  // max word num in a batch
  batch_id_buffer_ = batch_id_host_.get();

  label_host_.reset(new int[10240],
                    [](int *p) {delete[] p;});    // max label in a batch
  label_ptr_ = label_host_.get();
}

  // todo: use elegant implemention for this function
bool TextClassDataFeed::ReadBatch() {
  paddle::framework::Vector<size_t> offset;
  int tlen = 0;
  int llen = 0;
  int inst_idx = 0;
  offset.resize(batch_size_ + 1);
  offset[0] = 0;
  while (inst_idx < batch_size_) {
    int ptr_offset = 0;
    if (file_content_buffer_ptr_ - file_content_buffer_ >= file_size_) {
      break;
    }

    memcpy(reinterpret_cast<char *>(&llen),
          file_content_buffer_ptr_ + ptr_offset,
          sizeof(int));
    ptr_offset += sizeof(int);

    memcpy(reinterpret_cast<char *>(batch_id_buffer_ + tlen),
          file_content_buffer_ptr_ + ptr_offset,
          llen * sizeof(int));
    tlen += llen;

    offset[inst_idx + 1] = offset[inst_idx] + llen;
    ptr_offset += sizeof(int) * llen;

    memcpy(reinterpret_cast<char *>(label_ptr_ + inst_idx),
          file_content_buffer_ptr_ + ptr_offset,
          sizeof(int));
    ptr_offset += sizeof(int);

    file_content_buffer_ptr_ += ptr_offset;
    inst_idx++;
  }

  if (inst_idx != batch_size_) {
    return false;
  }

  LoD input_lod{offset};
  paddle::framework::Vector<size_t> label_offset;
  label_offset.resize(batch_size_ + 1);
  for (int i = 0; i <= batch_size_; ++i) {
    label_offset[i] = i;
  }

  LoD label_lod{label_offset};
  int64_t* input_ptr = feed_vec_[0]->mutable_data<int64_t>(
      {static_cast<int64_t>(offset.back()), 1},
      platform::CPUPlace());
  int64_t* label_ptr = feed_vec_[1]->mutable_data<int64_t>({batch_size_, 1},
                                                          platform::CPUPlace());
  for (unsigned int i = 0; i < offset.back(); ++i) {
    input_ptr[i] = static_cast<int64_t>(batch_id_buffer_[i]);
  }
  for (int i = 0; i < batch_size_; ++i) {
    label_ptr[i] = static_cast<int64_t>(label_ptr_[i]);
  }
  feed_vec_[0]->set_lod(input_lod);
  feed_vec_[1]->set_lod(label_lod);
  return true;
}

void TextClassDataFeed::AddFeedVar(Variable* feed, const std::string& name) {
  for (unsigned int i = 0; i < use_slot_alias_.size(); ++i) {
    if (name == use_slot_alias_[i]) {
      feed_vec_[i] = feed->GetMutable<LoDTensor>();
    }
  }
}

bool TextClassDataFeed::SetFile(const char* filename) {
  // termnum termid termid ... termid label
  int filesize = ReadWholeFile(filename, file_content_buffer_);
  // todo , remove magic number
  if (filesize < 0 || filesize >= 1024 * 1024 * 1024) {
    return false;
  }
  file_content_buffer_ptr_ = file_content_buffer_;
  file_size_ = filesize;
  return true;
}

int TextClassDataFeed::ReadWholeFile(const std::string& filename,
                                       char* buffer) {
  std::ifstream ifs(filename.c_str(), std::ios::binary);
  if (ifs.fail()) {
    return -1;
  }

  ifs.seekg(0, std::ios::end);
  int file_size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  ifs.read(buffer, file_size);
  return file_size;
}

}   // namespace framework
}   // namespace paddle
/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */

