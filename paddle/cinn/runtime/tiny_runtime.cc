// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <dlfcn.h>
#include <omp.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "paddle/cinn/runtime/cinn_runtime.h"

extern "C" {
int max_num_workers = std::thread::hardware_concurrency();
// move to standalone file
struct param_context_t {
  int major_v;
  int minor_v;
  std::vector<uint8_t> buf;
  std::vector<std::vector<uint8_t>> temporary;
  std::map<std::string, cinn_pod_value_t> name2podvalue;
  std::vector<std::string> instructions;
  std::vector<int> inst_argc;
  std::vector<cinn_pod_value_t *> inst_argv;
};

void *load_program(const char *paramfile) {
  FILE *f = fopen(paramfile, "r");
  fseek(f, 0, SEEK_END);
  int fsize = ftell(f);
  rewind(f);
  if (fsize < 32) {
    fclose(f);
    return nullptr;
  }

  std::unique_ptr<param_context_t> ctx(new param_context_t{});
  int alignment = std::max(alignof(cinn_pod_value_t), alignof(cinn_buffer_t));
  ctx->buf.resize(fsize + alignment);
  uint8_t *buf = ctx->buf.data();
  if ((uintptr_t)buf % alignment) {
    buf = buf + alignment - ((uintptr_t)buf % alignment);
  }
  fread(buf, 1, fsize, f);
  fclose(f);

  if (std::string(buf, buf + 4) != "CINN") {
    // TODO(hp03): LOG fatal
    return nullptr;
  }
  // TODO(hp03): check param file version
  ctx->major_v = *reinterpret_cast<int *>(buf + 4);
  ctx->minor_v = *reinterpret_cast<int *>(buf + 8);

  int *namelist_pos = reinterpret_cast<int *>(buf + 16);
  int *podvalue_pos = reinterpret_cast<int *>(buf + *namelist_pos);
  int *persistent_pos = reinterpret_cast<int *>(buf + *podvalue_pos);
  int *inst_pos = reinterpret_cast<int *>(buf + *persistent_pos);
  if (fsize < *inst_pos) {
    return nullptr;
  }

  int namelen = namelist_pos[1];
  std::vector<const char *> namev(namelen);
  std::map<std::string, int> name2index;
  for (int i = 0; i < namelen; i++) {
    int offset = (namelist_pos + 2)[i];
    namev[i] = reinterpret_cast<char *>(buf + offset);
    name2index[namev[i]] = i;
  }

  cinn_buffer_t *cb = reinterpret_cast<cinn_buffer_t *>(buf + podvalue_pos[1]);
  for (int i = 0; i < namelen; i++) {
    // currently only CPU device is supported, so just use malloc
    if (cb[i].memory) {
      cb[i].memory = buf + (uintptr_t)cb[i].memory;
    } else {
      int alignment = cb[i].align;
      if (alignment == 0) {
        alignment = 4;
      }
      ctx->temporary.emplace_back(alignment + cb[i].memory_size);
      uint8_t *tbuf = ctx->temporary.back().data();
      if ((uintptr_t)tbuf % alignment) {
        tbuf = tbuf + alignment - ((uintptr_t)tbuf % alignment);
      }
      cb[i].memory = tbuf;
    }
    ctx->name2podvalue[namev[i]] = cinn_pod_value_t(cb + i);
  }
  for (int i = 0; i < inst_pos[1]; i++) {
    const char *inst = (const char *)(buf + inst_pos[2 + i * 3 + 0]);
    ctx->instructions.push_back(inst);
    int instargc = inst_pos[2 + i * 3 + 1];
    ctx->inst_argc.push_back(instargc);
    cinn_pod_value_t *argv =
        reinterpret_cast<cinn_pod_value_t *>(buf + inst_pos[2 + i * 3 + 2]);
    for (int i = 0; i < instargc; i++) {
      int idx = (uintptr_t)((cinn_buffer_t *)(argv[i]));  // NOLINT
      cinn_value_t tmp_v;
      tmp_v.v_handle = &cb[idx];
      argv[i].set_value(tmp_v);
    }
    ctx->inst_argv.push_back(argv);
  }
  return ctx.release();
}

int set_maxconcurrency(int c) {
  int old_c = max_num_workers;
  max_num_workers = c;
  return old_c;
}

typedef void (*func_t)(cinn_pod_value_t *, int);
void run_program(void *ctx) {
  param_context_t *pc = reinterpret_cast<param_context_t *>(ctx);
  for (int i = 0; i < pc->instructions.size(); i++) {
    const char *sym = pc->instructions[i].c_str();
    void *p = dlsym(RTLD_DEFAULT, sym);
    func_t f = (func_t)p;
    f(pc->inst_argv[i], pc->inst_argc[i]);
  }
}

cinn_pod_value_t *get_pod_value(void *ctx, const char *tname) {
  param_context_t *pc = reinterpret_cast<param_context_t *>(ctx);
  if (pc->name2podvalue.find(tname) != pc->name2podvalue.end()) {
    return &pc->name2podvalue[tname];
  }
  return nullptr;
}

typedef int (*FCINNParallelLambda)(int task_id, int num_task, void *datas);
int cinn_backend_parallel_launch(FCINNParallelLambda flambda,
                                 void *datas,
                                 int num_task) {
  int num_workers = max_num_workers;
  if (num_task == 0) num_task = num_workers;
  omp_set_num_threads(num_task);
#pragma omp parallel num_threads(num_task)
  {
    int thread_num = omp_get_thread_num();
    (*flambda)(thread_num, num_task, datas);
  }
  return 0;
}
}
