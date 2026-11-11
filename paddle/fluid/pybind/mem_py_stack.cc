// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pybind/mem_py_stack.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace paddle {
namespace pybind {

namespace {

// One resolved-at-snapshot frame: the (interned, alive) code object plus the
// line number captured at dispatch time. We deliberately do NOT build strings
// on the hot path.
struct Frame {
  PyCodeObject* code;  // borrowed here; kept alive by interned_ (INCREF'd once)
  int line;
};

// Upper bound on distinct stacks kept, to bound host memory. Once reached,
// CaptureCurrentPyStack returns 0 (falls back to op_name) instead of interning
// more.
constexpr size_t kMaxUniqueStacks = 200000;

// Whether frame walking/interning is active. Independent from recording being
// enabled: both must be true to capture.
std::atomic<bool> g_capture_enabled{false};

std::mutex g_mu;
// stack_id (1-based) -> frame vector. Index 0 of the vector is stack_id 1.
std::vector<std::vector<Frame>> g_stacks;
// hash -> candidate stack_ids (collisions resolved by full frame compare).
std::unordered_map<uint64_t, std::vector<uint64_t>> g_hash_index;
// Unique code objects held alive (one INCREF each) so raw pointers stay valid
// as map keys / frame refs for the whole recording session.
std::unordered_set<PyCodeObject*> g_interned;

uint64_t HashFrames(const std::vector<Frame>& frames) {
  // FNV-1a over (code pointer, line) pairs.
  uint64_t h = 1469598103934665603ULL;
  auto mix = [&h](uint64_t v) {
    for (int i = 0; i < 8; ++i) {
      h ^= (v & 0xff);
      h *= 1099511628211ULL;
      v >>= 8;
    }
  };
  for (const auto& f : frames) {
    mix(reinterpret_cast<uint64_t>(f.code));
    mix(static_cast<uint64_t>(static_cast<uint32_t>(f.line)));
  }
  return h;
}

bool FramesEqual(const std::vector<Frame>& a, const std::vector<Frame>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].code != b[i].code || a[i].line != b[i].line) return false;
  }
  return true;
}

}  // namespace

void SetMemPyStackCaptureEnabled(bool enabled) {
  g_capture_enabled.store(enabled, std::memory_order_relaxed);
}

MemStackCapture CaptureCurrentPyStack() {
  const bool active = paddle::memory::MemHistoryEnabled();
  if (!active) return {false, 0};
  if (!g_capture_enabled.load(std::memory_order_relaxed)) return {true, 0};

  // Caller holds the GIL. Walk frames innermost -> outermost. PyFrame_GetCode
  // and PyFrame_GetBack return new references; we release the temporary frame /
  // code references at the end and separately INCREF codes that we newly
  // intern.
  std::vector<Frame> frames;
  std::vector<PyCodeObject*> temp_code_refs;
  PyFrameObject* f = PyEval_GetFrame();  // borrowed
  Py_XINCREF(f);                         // own our iteration reference
  while (f != nullptr) {
    int line = PyFrame_GetLineNumber(f);
    PyCodeObject* code = PyFrame_GetCode(f);  // new ref
    frames.push_back(Frame{code, line});
    temp_code_refs.push_back(code);
    PyFrameObject* back = PyFrame_GetBack(f);  // new ref
    Py_DECREF(f);
    f = back;
  }

  uint64_t result = 0;
  if (!frames.empty()) {
    uint64_t h = HashFrames(frames);
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_hash_index.find(h);
    if (it != g_hash_index.end()) {
      for (uint64_t sid : it->second) {
        if (FramesEqual(g_stacks[sid - 1], frames)) {
          result = sid;
          break;
        }
      }
    }
    if (result == 0 && g_stacks.size() < kMaxUniqueStacks) {
      // Intern any not-yet-seen code objects (one INCREF each, session-lived).
      for (const auto& fr : frames) {
        if (g_interned.insert(fr.code).second) {
          Py_INCREF(fr.code);
        }
      }
      g_stacks.push_back(frames);
      result = g_stacks.size();  // 1-based
      g_hash_index[h].push_back(result);
    }
  }

  // Release the temporary references obtained during the walk. Interned codes
  // survive via their separate INCREF above.
  for (PyCodeObject* code : temp_code_refs) {
    Py_DECREF(code);
  }
  return {true, result};
}

::pybind11::list ResolveStack(uint64_t stack_id) {
  namespace py = ::pybind11;
  py::list out;
  if (stack_id == 0) return out;
  std::lock_guard<std::mutex> lock(g_mu);
  if (stack_id > g_stacks.size()) return out;
  const auto& frames = g_stacks[stack_id - 1];
  for (const auto& fr : frames) {  // innermost first
    py::dict d;
    const char* fn = PyUnicode_AsUTF8(fr.code->co_filename);
    const char* nm = PyUnicode_AsUTF8(fr.code->co_name);
    d["filename"] = fn ? fn : "";
    d["name"] = nm ? nm : "";
    d["line"] = fr.line;
    out.append(d);
  }
  return out;
}

void ClearMemPyStacks() {
  // Caller holds the GIL, so Py_DECREF is safe.
  std::lock_guard<std::mutex> lock(g_mu);
  for (PyCodeObject* code : g_interned) {
    Py_DECREF(code);
  }
  g_interned.clear();
  g_stacks.clear();
  g_hash_index.clear();
}

}  // namespace pybind
}  // namespace paddle
