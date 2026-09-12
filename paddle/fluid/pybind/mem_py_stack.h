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

#pragma once

#include <cstddef>
#include <cstdint>

#include "paddle/phi/core/memory/allocation/memory_history_recorder.h"
#include "pybind11/pybind11.h"

// Python call-stack capture for the GPU memory-history recorder. All Python
// C-API access lives here so that phi core (the allocator hot path) stays free
// of any Python dependency. See design: capture at op dispatch while the GIL is
// still held, intern to an opaque stack_id, resolve to frames at snapshot time.

namespace paddle {
namespace pybind {

struct MemStackCapture {
  bool active;
  uint64_t stack_id;
};

// Capture the current Python call stack (innermost frame first). `active`
// reports whether memory history was enabled at the call site, while stack_id
// is 0 when stack capture is disabled or no stack was captured. CALLER MUST
// HOLD THE GIL. Called just before PyEval_SaveThread releases the GIL.
MemStackCapture CaptureCurrentPyStack();

// Resolve a previously captured stack_id into a list of frame dicts
// ``{filename, name, line}`` (innermost first). Empty list for id 0 or unknown.
// CALLER MUST HOLD THE GIL (invoked from the pybind snapshot builder).
::pybind11::list ResolveStack(uint64_t stack_id);

// Drop all interned stacks and Py_DECREF every interned code object. CALLER
// MUST HOLD THE GIL. Invoked when (re)configuring recording from Python.
void ClearMemPyStacks();

// Toggle whether CaptureCurrentPyStack actually walks/interns frames. When
// false, capture is a single atomic read returning 0.
void SetMemPyStackCaptureEnabled(bool enabled);

}  // namespace pybind
}  // namespace paddle
