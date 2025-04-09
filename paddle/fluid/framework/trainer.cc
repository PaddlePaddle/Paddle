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

#include "paddle/fluid/framework/trainer.h"

#include "io/fs.h"

namespace paddle::framework {

void TrainerBase::SetScope(Scope* root_scope) { root_scope_ = root_scope; }

void TrainerBase::ParseDumpConfig(const TrainerDesc& desc) {
  dump_fields_path_ = desc.dump_fields_path();
  need_dump_field_ = false;
  need_dump_param_ = false;
  dump_fields_mode_ = desc.dump_fields_mode();
  if (dump_fields_path_.empty()) {
    VLOG(2) << "dump_fields_path_ is empty";
    return;
  }
  auto& file_list = dataset_ptr_->GetFileList();
  if (file_list.empty()) {
    VLOG(2) << "file_list is empty";
    return;
  }

  dump_converter_ = desc.dump_converter();
  if (desc.dump_fields_size() != 0) {
    need_dump_field_ = true;
    dump_fields_.resize(desc.dump_fields_size());
    for (int i = 0; i < desc.dump_fields_size(); ++i) {
      dump_fields_[i] = desc.dump_fields(i);
    }
  }

  if (desc.dump_param_size() != 0) {
    need_dump_param_ = true;
    dump_param_.resize(desc.dump_param_size());
    for (int i = 0; i < desc.dump_param_size(); ++i) {
      dump_param_[i] = desc.dump_param(i);
    }
  }
}

void TrainerBase::DumpWork(int tid) {
#ifdef _LINUX
  int err_no = 0;
  // GetDumpPath is implemented in each Trainer
  std::string path = GetDumpPath(tid);
  std::shared_ptr<FILE> fp;
  if (dump_fields_mode_ == "a") {
    VLOG(3) << "dump field mode append";
    fp = fs_open_append_write(path, &err_no, dump_converter_);
  } else {
    VLOG(3) << "dump field mode overwrite";
    fp = fs_open_write(path, &err_no, dump_converter_);
  }
  while (true) {
    std::string out_str;
    if (!queue_->Get(out_str)) {
      break;
    }
    size_t write_count =
        fwrite_unlocked(out_str.data(), 1, out_str.length(), fp.get());
    if (write_count != out_str.length()) {
      VLOG(3) << "dump text failed";
      continue;
    }
    write_count = fwrite_unlocked("\n", 1, 1, fp.get());
    if (write_count != 1) {
      VLOG(3) << "dump text failed";
      continue;
    }
  }
#endif
}

void TrainerBase::FinalizeDumpEnv() {
  queue_->Close();
  for (auto& th : dump_thread_) {
    th.join();
  }
  queue_.reset();
}

}  // namespace paddle::framework
