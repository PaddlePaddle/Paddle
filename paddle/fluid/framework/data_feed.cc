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
void TextClassDataFeed::init(const datafeed::DataFeedParameter& feed_param) {
  // hard coding for a specific datafeed
  _feed_vec.resize(2);
  // _feed_vec[0].reset(new LoDTensor);
  // _feed_vec[1].reset(new LoDTensor);
  _all_slot_ids = {0, 1};
  _use_slot_ids = {0, 1};
  _use_slot_alias = {"words", "label"};
  _file_content_buffer_host.reset(new char[200*1024*1024],
                                  [](char *p) {delete[] p;});
  _file_content_buffer = _file_content_buffer_host.get();
  _file_content_buffer_ptr = _file_content_buffer;
  _batch_id_host.reset(new int[10240*1024],
                      [](int *p) {delete[] p;});  // max word num in a batch
  _label_host.reset(new int[10240],
                    [](int *p) {delete[] p;});    // max label in a batch
  _batch_id_buffer = _batch_id_host.get();
  _label_ptr = _label_host.get();
}

  // todo: use elegant implemention for this function
bool TextClassDataFeed::read_batch() {
  paddle::framework::Vector<size_t> offset;
  int tlen = 0;
  int llen = 0;
  int inst_idx = 0;
  offset.resize(_batch_size + 1);
  offset[0] = 0;
  while (inst_idx < _batch_size) {
    int ptr_offset = 0;
    if (_file_content_buffer_ptr - _file_content_buffer >= _file_size) {
      break;
    }

    memcpy(reinterpret_cast<char *>(&llen),
          _file_content_buffer_ptr + ptr_offset,
          sizeof(int));
    ptr_offset += sizeof(int);

    memcpy(reinterpret_cast<char *>(_batch_id_buffer + tlen),
          _file_content_buffer_ptr + ptr_offset,
          llen * sizeof(int));
    tlen += llen;

    offset[inst_idx + 1] = offset[inst_idx] + llen;
    ptr_offset += sizeof(int) * llen;

    memcpy(reinterpret_cast<char *>(_label_ptr + inst_idx),
          _file_content_buffer_ptr + ptr_offset,
          sizeof(int));
    ptr_offset += sizeof(int);

    _file_content_buffer_ptr += ptr_offset;
    inst_idx++;
  }

  if (inst_idx != _batch_size) {
    return false;
  }

  LoD input_lod{offset};
  paddle::framework::Vector<size_t> label_offset;
  label_offset.resize(_batch_size + 1);
  for (int i = 0; i <= _batch_size; ++i) {
    label_offset[i] = i;
  }

  LoD label_lod{label_offset};
  int64_t* input_ptr = _feed_vec[0]->mutable_data<int64_t>(
      {static_cast<int64_t>(offset.back()), 1},
      platform::CPUPlace());
  int64_t* label_ptr = _feed_vec[1]->mutable_data<int64_t>({_batch_size, 1},
                                                          platform::CPUPlace());
  for (unsigned int i = 0; i < offset.back(); ++i) {
    input_ptr[i] = static_cast<int64_t>(_batch_id_buffer[i]);
  }
  for (int i = 0; i < _batch_size; ++i) {
    label_ptr[i] = static_cast<int64_t>(_label_ptr[i]);
  }
  _feed_vec[0]->set_lod(input_lod);
  _feed_vec[1]->set_lod(label_lod);
  return true;
}

void TextClassDataFeed::add_feed_var(Variable* feed, const std::string& name) {
  for (unsigned int i = 0; i < _use_slot_alias.size(); ++i) {
    if (name == _use_slot_alias[i]) {
      _feed_vec[i] = feed->GetMutable<LoDTensor>();
    }
  }
}

bool TextClassDataFeed::set_file(const char* filename) {
  // termnum termid termid ... termid label
  int filesize = read_whole_file(filename, _file_content_buffer);
  // todo , remove magic number
  if (filesize < 0 || filesize >= 1024 * 1024 * 1024) {
    return false;
  }
  _file_content_buffer_ptr = _file_content_buffer;
  _file_size = filesize;
  return true;
}

int TextClassDataFeed::read_whole_file(const std::string& filename,
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

