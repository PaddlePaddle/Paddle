// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/program.h"

namespace cinn {
namespace hlir {
namespace framework {

Program::Program(const std::shared_ptr<Scope>& scope,
                 std::vector<std::unique_ptr<Instruction>>&& instrs)
    : scope_(scope) {
  for (auto& ins : instrs) {
    if (ins->pre_run) {
      prerun_instrs_.push_back(std::move(ins));
    } else {
      instrs_.push_back(std::move(ins));
    }
  }
}

void Program::PreRun(
    const std::map<std::string, cinn_pod_value_t>* name2podargs) {
  for (auto& ins : prerun_instrs_) {
    ins->Run(name2podargs);
  }
  for (auto& ins : instrs_) {
    if (ins->size() == 4) {
      ins->PreRun(name2podargs);
    }
  }
}

void Program::Export(const std::vector<std::string>& persistent_vars,
                     const std::string& filename) {
  auto writeplaceholder = [=](int s, int n, FILE* f) -> int {
    int pos = ftell(f);
    for (int i = 0; i < s * n; i++) {
      fwrite("\0", 1, 1, f);
    }
    return pos;
  };
  auto setplaceholder = [=](int p, void* b, int s, int n, FILE* f) {
    int cur = ftell(f);
    fseek(f, p, SEEK_SET);
    fwrite(b, s, n, f);
    fseek(f, cur, SEEK_SET);
  };
  auto tellplaceholder = [=](int p, FILE* f) {
    int cur = ftell(f);
    setplaceholder(p, &cur, 4, 1, f);
  };
  auto padding = [=](int alignment, uint8_t value, FILE* f) {
    int cur = ftell(f);
    int padding = (alignment - (cur % alignment)) % alignment;
    for (int i = 0; i < padding; i++) {
      fwrite(&value, 1, 1, f);
    }
  };
  auto varnames = scope_->var_names();
  std::unordered_map<std::string, int> varindex;
  for (int i = 0; i < varnames.size(); i++) {
    varindex[(std::string)varnames[i]] = i;
  }

  FILE* f = fopen(filename.c_str(), "w+");

  fwrite("CINN", 4, 1, f);
  int major_v = 0;
  int minor_v = 0;
  fwrite(&major_v, 4, 1, f);
  fwrite(&minor_v, 4, 1, f);
  int unused_v = 0;
  fwrite(&unused_v, 4, 1, f);

  // varname list
  int varnamesec = writeplaceholder(4, 1, f);
  int namesnum = varnames.size();
  fwrite(&namesnum, 4, 1, f);
  int nameoffset = writeplaceholder(4, namesnum, f);
  for (int i = 0; i < namesnum; i++) {
    int namelen = varnames[i].size();
    fwrite(&namelen, 4, 1, f);
    tellplaceholder(nameoffset + i * 4, f);
    fwrite(varnames[i].data(), namelen, 1, f);
    fwrite("\0", 1, 1, f);
  }
  padding(16, 0, f);
  tellplaceholder(varnamesec, f);
  // pod_values
  int buffersec = writeplaceholder(4, 1, f);
  int bufoffset = writeplaceholder(4, 1, f);
  padding(alignof(cinn_buffer_t), 0, f);
  tellplaceholder(bufoffset, f);
  std::vector<std::pair<cinn_buffer_t*, int>> pvars;
  for (auto& varname : varnames) {
    std::string name = (std::string)varname;
    auto t = scope_->GetTensor(name);
    cinn_buffer_t buffer = *t->buffer();
    buffer.memory = reinterpret_cast<uint8_t*>(0);
    if (std::find(persistent_vars.begin(), persistent_vars.end(), name) !=
        persistent_vars.end()) {
      pvars.emplace_back(t->buffer(),
                         ftell(f) + offsetof(cinn_buffer_t, memory));
    }
    fwrite(&buffer, sizeof(cinn_buffer_t), 1, f);
  }
  padding(16, 0, f);
  tellplaceholder(buffersec, f);
  // persistent_buffers
  int pbuffer = writeplaceholder(4, 1, f);
  for (auto& p : pvars) {
    if (p.first->align) {
      padding(p.first->align, 0, f);
    }
    tellplaceholder(p.second, f);
    fwrite(p.first->memory, p.first->memory_size, 1, f);
  }
  padding(16, 0, f);
  tellplaceholder(pbuffer, f);
  // instructions
  int instsec = writeplaceholder(4, 1, f);
  int insnum = 0;
  for (auto& ins : instrs_) {
    ins->Run(nullptr, true);
    insnum += ins->GetFnNames().size();
  }
  fwrite(&insnum, 4, 1, f);
  int instplaceholder = writeplaceholder(4 * 3, insnum, f);
  int findex = 0;
  for (auto& ins : instrs_) {
    auto& in_args = ins->GetInArgs();
    auto& out_args = ins->GetOutArgs();
    auto& fn_names = ins->GetFnNames();
    for (int i = 0; i < fn_names.size(); i++, findex++) {
      std::vector<std::string> all_args(in_args[i].begin(), in_args[i].end());
      all_args.insert(
          std::end(all_args), out_args[i].begin(), out_args[i].end());
      auto fname = fn_names[i];
      int fnamesize = fname.size();
      fwrite(&fnamesize, 4, 1, f);
      tellplaceholder(instplaceholder + findex * 12, f);
      fwrite(fname.c_str(), fname.size(), 1, f);
      fwrite("\0", 1, 1, f);
      int argsize = all_args.size();
      setplaceholder(instplaceholder + findex * 12 + 4, &argsize, 4, 1, f);
      padding(alignof(cinn_pod_value_t), 0, f);
      tellplaceholder(instplaceholder + findex * 12 + 8, f);
      for (auto& arg : all_args) {
        uintptr_t bufindex = varindex[arg];
        cinn_pod_value_t v(reinterpret_cast<cinn_buffer_t*>(bufindex));
        fwrite(&v, sizeof(cinn_pod_value_t), 1, f);
      }
    }
  }
  padding(16, 0, f);
  tellplaceholder(instsec, f);
  fclose(f);
}

void Program::Execute(
    const std::map<std::string, cinn_pod_value_t>* name2podargs,
    void* stream,
    bool use_cache) {
  for (auto& ins : instrs_) {
    ins->Run(name2podargs, false, stream, use_cache);
  }
#ifdef CINN_WITH_CUDA
  VLOG(4) << "-- The value of the used stream: " << stream;
  if (instrs_[0]->target_.arch == Target::Arch::NVGPU && stream == nullptr) {
    CUDA_CALL(cudaDeviceSynchronize());
  }
#endif
}

void Program::ExecuteTest(int repeat_) {
  cinn::utils::Timer timer1;
  for (int i = 0; i < 100; i++) {
    for (auto& ins : instrs_) {
      ins->Run();
    }
  }
  timer1.Start();
  for (int i = 0; i < repeat_; i++) {
    for (auto& ins : instrs_) {
      ins->Run();
    }
  }
#ifdef CINN_WITH_CUDA
  if (instrs_[0]->target_.arch == Target::Arch::NVGPU) {
    CUDA_CALL(cudaDeviceSynchronize());
  }
#endif
  double test_op_time = timer1.Stop() / repeat_;
  VLOG(3) << "Repeat times: [" << repeat_ << "], average op time: ["
          << test_op_time << "] ms";
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
