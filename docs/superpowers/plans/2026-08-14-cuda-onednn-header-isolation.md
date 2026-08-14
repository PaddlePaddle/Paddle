# CUDA oneDNN Header Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent CUDA translation units from parsing oneDNN C++ headers while preserving existing oneDNN tensor descriptor behavior.

**Architecture:** Keep `DenseTensor` and common context headers backend-neutral. Move `OneDNNStorageProperties` and descriptor access into the existing oneDNN backend, then replace the old `DenseTensor::mem_desc()` API with backend-local free functions. Common headers retain only declarations that are safe with incomplete types; host oneDNN consumers include their backend dependencies explicitly.

**Tech Stack:** C++20, Paddle PHI, oneDNN C++ API, CMake/Ninja, clang preprocessor, ast-grep.

## Global Constraints

- Base branch and commit: `origin/develop` at `16037ff1effb88625041f9a1c540e8b2af3ab5c1`.
- Working branch: `codex/cuda-cxx20-onednn-isolation`.
- Do not update or patch oneDNN.
- Do not require NVCC 13.1; the target failing toolchain is CUDA 12.9.41 with GCC 13.3 and C++20.
- Do not add a forced-include header or any specialization in namespace `std`.
- Preserve oneDNN descriptor ownership, `format`, layout changes, and invalid-property error behavior.
- Preserve the pre-existing work saved in `stash@{0}`; do not apply or drop it during this change.
- Do not relocate `paddle/phi/kernels/funcs/data_layout_transform.h`: its direct consumers are host/oneDNN sources. Stop re-exporting it from the common Fluid layout-transform header instead.

---

### Task 1: Establish the failing public-header isolation check

**Files:**
- Inspect: `paddle/phi/core/dense_tensor.h`
- Inspect: `paddle/phi/core/platform/device_context.h`
- Inspect: `paddle/phi/backends/all_context.h`

**Interfaces:**
- Consumes: public Paddle headers with `PADDLE_WITH_DNNL` enabled.
- Produces: a repeatable red/green command proving whether a public header reaches `dnnl.hpp`.

- [ ] **Step 1: Run the preprocessor check against the unmodified implementation**

```bash
headers=(
  paddle/phi/core/dense_tensor.h
  paddle/phi/core/platform/device_context.h
  paddle/phi/backends/all_context.h
)
for header in "${headers[@]}"; do
  trace="/tmp/$(basename "$header").include-trace"
  printf '#include "%s"\n' "$header" |
    clang++ -std=c++20 -DPADDLE_WITH_DNNL \
      -I. -Ithird_party/onednn/include -E -H -x c++ - \
      >/dev/null 2>"$trace"
  if rg -n '(^|/)dnnl\.hpp' "$trace"; then
    echo "FAIL: $header reaches dnnl.hpp"
  else
    echo "PASS: $header is isolated"
  fi
done
```

Expected before implementation: all three headers print `FAIL` and their include traces contain `dnnl.hpp`.

- [ ] **Step 2: Record the exact include chains used by the fix**

```bash
rg -n -C 4 'dnnl\.hpp' \
  /tmp/dense_tensor.h.include-trace \
  /tmp/device_context.h.include-trace \
  /tmp/all_context.h.include-trace
```

Expected: the traces identify `storage_properties.h`, `device_context.h`, and `onednn_context.h`/`all_context.h` as the direct common-header entry points.

---

### Task 2: Move oneDNN storage properties and descriptor access into the backend

**Files:**
- Create: `paddle/phi/backends/onednn/onednn_storage_properties.h`
- Modify: `paddle/phi/backends/onednn/onednn_helper.h`
- Modify: `paddle/phi/core/storage_properties.h`
- Modify: `paddle/phi/core/storage_properties.cc`
- Modify: `paddle/phi/core/dense_tensor.inl`
- Modify: `paddle/phi/core/dense_tensor_impl.cc`
- Modify: `paddle/phi/core/dense_tensor.cc`
- Modify: `paddle/phi/core/utils/type_info.cc`
- Modify: `test/cpp/phi/core/test_dense_tensor.cc`

**Interfaces:**
- Consumes: `DenseTensor::storage_properties_initialized()`, `DenseTensor::storage_properties<T>()`, `DenseTensor::set_storage_properties()`, and `DenseTensor::set_layout()`.
- Produces: `phi::OneDNNStorageProperties`, `phi::funcs::GetOneDNNMemDesc(const DenseTensor&)`, and `phi::funcs::SetOneDNNMemDesc(DenseTensor*, const dnnl::memory::desc&)`.

- [ ] **Step 1: Add the backend-local storage property type**

Create `paddle/phi/backends/onednn/onednn_storage_properties.h` with the existing fields and type registration unchanged:

```cpp
#pragma once

#ifdef PADDLE_WITH_DNNL
#include "dnnl.hpp"  // NOLINT

#include "paddle/phi/core/storage_properties.h"

namespace phi {

struct OneDNNStorageProperties
    : public StorageProperties,
      public TypeInfoTraits<StorageProperties, OneDNNStorageProperties> {
  ~OneDNNStorageProperties() override = default;
  static const char* name() { return "OneDNNStorageProperties"; }

  dnnl::memory::format_tag format = dnnl::memory::format_tag::undef;
  dnnl::memory::desc mem_desc;
};

}  // namespace phi
#endif
```

- [ ] **Step 2: Make the core storage-property header backend-neutral**

Delete the guarded `dnnl.hpp` include and the complete `OneDNNStorageProperties` definition from `paddle/phi/core/storage_properties.h`. Leave `StorageProperties`, NPU, and XPU behavior unchanged.

- [ ] **Step 3: Include the backend type only in host implementation and test files that need it**

Under `PADDLE_WITH_DNNL`, include `paddle/phi/backends/onednn/onednn_storage_properties.h` from:

```text
paddle/phi/core/storage_properties.cc
paddle/phi/core/dense_tensor.cc
paddle/phi/core/utils/type_info.cc
test/cpp/phi/core/test_dense_tensor.cc
```

Keep the existing `CopyStorageProperties`, explicit `storage_properties<T>()` instantiation, runtime type check, and unit-test behavior unchanged.

- [ ] **Step 4: Add backend-local descriptor helpers**

Include `onednn_storage_properties.h` from `onednn_helper.h`, then add these inline functions in `phi::funcs`:

```cpp
inline const dnnl::memory::desc& GetOneDNNMemDesc(
    const DenseTensor& tensor) {
  if (!tensor.storage_properties_initialized()) {
    static const dnnl::memory::desc undef_desc;
    return undef_desc;
  }
  return tensor.storage_properties<OneDNNStorageProperties>().mem_desc;
}

inline void SetOneDNNMemDesc(DenseTensor* tensor,
                             const dnnl::memory::desc& mem_desc) {
  auto properties = std::make_unique<OneDNNStorageProperties>();
  if (tensor->storage_properties_initialized()) {
    properties->format =
        tensor->storage_properties<OneDNNStorageProperties>().format;
  }
  properties->mem_desc = mem_desc;
  tensor->set_storage_properties(std::move(properties));
  tensor->set_layout(DataLayout::ONEDNN);
}
```

The setter replaces the `unique_ptr` because core exposes no mutable backend property API; copying `format` preserves the old in-place update semantics.

- [ ] **Step 5: Remove the oneDNN-specific `DenseTensor` members**

Delete the guarded declarations from `paddle/phi/core/dense_tensor.inl` and the guarded definitions from `paddle/phi/core/dense_tensor_impl.cc`:

```cpp
const dnnl::memory::desc& mem_desc() const;
void set_mem_desc(const dnnl::memory::desc& mem_desc);
```

- [ ] **Step 6: Format and commit the backend boundary**

```bash
pre-commit run clang-format --files \
  paddle/phi/backends/onednn/onednn_storage_properties.h \
  paddle/phi/backends/onednn/onednn_helper.h \
  paddle/phi/core/storage_properties.h \
  paddle/phi/core/storage_properties.cc \
  paddle/phi/core/dense_tensor.inl \
  paddle/phi/core/dense_tensor_impl.cc \
  paddle/phi/core/dense_tensor.cc \
  paddle/phi/core/utils/type_info.cc \
  test/cpp/phi/core/test_dense_tensor.cc
git diff --check
git add \
  paddle/phi/backends/onednn/onednn_storage_properties.h \
  paddle/phi/backends/onednn/onednn_helper.h \
  paddle/phi/core/storage_properties.h \
  paddle/phi/core/storage_properties.cc \
  paddle/phi/core/dense_tensor.inl \
  paddle/phi/core/dense_tensor_impl.cc \
  paddle/phi/core/dense_tensor.cc \
  paddle/phi/core/utils/type_info.cc \
  test/cpp/phi/core/test_dense_tensor.cc
git commit -m "refactor: isolate oneDNN storage properties"
```

---

### Task 3: Replace all descriptor member calls with backend helpers

**Files:**
- Modify: every path returned by the following fixed call-site inventory:

```text
paddle/fluid/framework/data_transform.cc
paddle/fluid/framework/new_executor/instruction/onednn/onednn_instruction.cc
paddle/fluid/framework/new_executor/instruction/onednn/onednn_legacy_instruction.cc
paddle/fluid/framework/new_executor/instruction/onednn/onednn_mixed_instruction.cc
paddle/fluid/framework/tensor_util.cc
paddle/fluid/operators/generator/get_expected_kernel_func.cc
paddle/fluid/operators/slice_op.cc
paddle/fluid/operators/split_op.cc
paddle/phi/backends/onednn/matmul_utils.h
paddle/phi/backends/onednn/onednn_reuse.h
paddle/phi/kernels/funcs/data_layout_transform.cc
paddle/phi/kernels/fusion/onednn/fc_kernel.cc
paddle/phi/kernels/fusion/onednn/fused_elementwise_kernel.cc
paddle/phi/kernels/fusion/onednn/fused_matmul_kernel.cc
paddle/phi/kernels/fusion/onednn/fused_softplus_kernel.cc
paddle/phi/kernels/fusion/onednn/fused_transpose_kernel.cc
paddle/phi/kernels/onednn/activation_grad_kernel.cc
paddle/phi/kernels/onednn/activation_kernel.cc
paddle/phi/kernels/onednn/add_n_kernel.cc
paddle/phi/kernels/onednn/batch_norm_grad_kernel.cc
paddle/phi/kernels/onednn/batch_norm_kernel.cc
paddle/phi/kernels/onednn/cast_kernel.cc
paddle/phi/kernels/onednn/clip_grad_kernel.cc
paddle/phi/kernels/onednn/clip_kernel.cc
paddle/phi/kernels/onednn/concat_grad_kernel.cc
paddle/phi/kernels/onednn/concat_kernel.cc
paddle/phi/kernels/onednn/conv_function.h
paddle/phi/kernels/onednn/conv_grad_kernel.cc
paddle/phi/kernels/onednn/conv_handler.h
paddle/phi/kernels/onednn/conv_transpose_kernel.cc
paddle/phi/kernels/onednn/dequantize_kernel.cc
paddle/phi/kernels/onednn/elementwise_grad_kernel.cc
paddle/phi/kernels/onednn/elementwise_kernel.cc
paddle/phi/kernels/onednn/expand_grad_kernel.cc
paddle/phi/kernels/onednn/expand_kernel.cc
paddle/phi/kernels/onednn/flatten_grad_kernel.cc
paddle/phi/kernels/onednn/flatten_kernel.cc
paddle/phi/kernels/onednn/full_kernel.cc
paddle/phi/kernels/onednn/gaussian_kernel.cc
paddle/phi/kernels/onednn/interpolate_kernel.cc
paddle/phi/kernels/onednn/layer_norm_kernel.cc
paddle/phi/kernels/onednn/log_softmax_kernel.cc
paddle/phi/kernels/onednn/lrn_kernel_impl.h
paddle/phi/kernels/onednn/matmul_grad_kernel.cc
paddle/phi/kernels/onednn/matmul_kernel.cc
paddle/phi/kernels/onednn/pad_kernel_impl.h
paddle/phi/kernels/onednn/pad3d_kernel.cc
paddle/phi/kernels/onednn/pool_grad_kernel.cc
paddle/phi/kernels/onednn/pool_kernel.cc
paddle/phi/kernels/onednn/prelu_grad_kernel.cc
paddle/phi/kernels/onednn/prelu_kernel.cc
paddle/phi/kernels/onednn/quantize_kernel.cc
paddle/phi/kernels/onednn/reduce_kernel_impl.h
paddle/phi/kernels/onednn/requantize_kernel.cc
paddle/phi/kernels/onednn/reshape_grad_kernel.cc
paddle/phi/kernels/onednn/reshape_kernel.cc
paddle/phi/kernels/onednn/scale_kernel.cc
paddle/phi/kernels/onednn/shape_kernel.cc
paddle/phi/kernels/onednn/shuffle_channel_kernel.cc
paddle/phi/kernels/onednn/slice_grad_kernel.cc
paddle/phi/kernels/onednn/slice_kernel.cc
paddle/phi/kernels/onednn/softmax_grad_kernel.cc
paddle/phi/kernels/onednn/softmax_kernel.cc
paddle/phi/kernels/onednn/softplus_kernel.cc
paddle/phi/kernels/onednn/split_kernel.cc
paddle/phi/kernels/onednn/squeeze_grad_kernel.cc
paddle/phi/kernels/onednn/squeeze_kernel.cc
paddle/phi/kernels/onednn/stack_kernel.cc
paddle/phi/kernels/onednn/transpose_grad_kernel.cc
paddle/phi/kernels/onednn/transpose_kernel.cc
paddle/phi/kernels/transfer_layout_kernel.cc
```

**Interfaces:**
- Consumes: the helpers created in Task 2.
- Produces: no remaining call to `DenseTensor::mem_desc()` or `DenseTensor::set_mem_desc()`.

- [ ] **Step 1: Verify ast-grep patterns against representative snippets**

```bash
printf '%s\n' \
  'auto a = tensor.mem_desc();' \
  'auto b = tensor_ptr->mem_desc();' \
  'tensor.set_mem_desc(desc);' \
  'tensor_ptr->set_mem_desc(desc);' |
  ast-grep run --pattern '$OBJ.mem_desc()' --lang cpp --stdin
```

Expected: the dot-access getter is matched without matching the arrow form or setters. Repeat for `$PTR->mem_desc()`, `$OBJ.set_mem_desc($DESC)`, and `$PTR->set_mem_desc($DESC)` before applying rewrites.

- [ ] **Step 2: Apply the four structural rewrites**

```bash
ast-grep run --pattern '$OBJ.mem_desc()' \
  --rewrite 'phi::funcs::GetOneDNNMemDesc($OBJ)' --lang cpp -U paddle
ast-grep run --pattern '$PTR->mem_desc()' \
  --rewrite 'phi::funcs::GetOneDNNMemDesc(*$PTR)' --lang cpp -U paddle
ast-grep run --pattern '$OBJ.set_mem_desc($DESC)' \
  --rewrite 'phi::funcs::SetOneDNNMemDesc(&($OBJ), $DESC)' --lang cpp -U paddle
ast-grep run --pattern '$PTR->set_mem_desc($DESC)' \
  --rewrite 'phi::funcs::SetOneDNNMemDesc($PTR, $DESC)' --lang cpp -U paddle
```

Review every rewrite involving a non-identifier receiver. Parenthesize dereferenced or address-taken expressions so evaluation count remains one.

- [ ] **Step 3: Add explicit helper includes where compilation no longer receives the declaration transitively**

Add this include to each changed file that does not already receive it from `onednn_reuse.h`, `conv_handler.h`, or another direct oneDNN backend header:

```cpp
#include "paddle/phi/backends/onednn/onednn_helper.h"
```

Use this check after edits:

```bash
rg -l 'GetOneDNNMemDesc|SetOneDNNMemDesc' paddle |
  xargs rg --files-without-match \
    'paddle/phi/backends/onednn/(onednn_helper|onednn_reuse|conv_handler)\.h'
```

For every returned file, add a direct `onednn_helper.h` include; do not expose the helper through a common core header.

- [ ] **Step 4: Prove the old member API is gone**

```bash
rg -n '(->|\.)mem_desc\(|set_mem_desc\(' paddle test
```

Expected: no matches. `GetOneDNNMemDesc` and `SetOneDNNMemDesc` are intentionally not matched by this expression.

- [ ] **Step 5: Format and commit the mechanical migration**

```bash
pre-commit run clang-format --files $(git diff --name-only --diff-filter=ACM | rg '\.(cc|h|inl)$')
git diff --check
git diff --name-only --diff-filter=ACM > /tmp/onednn-descriptor-callsite-files
git add --pathspec-from-file=/tmp/onednn-descriptor-callsite-files
git commit -m "refactor: use backend oneDNN descriptor helpers"
```

Before committing, inspect `git diff --cached --name-only` and unstage any path outside the fixed call-site inventory.

---

### Task 4: Remove the remaining common-header oneDNN leaks

**Files:**
- Modify: `paddle/phi/core/compat/convert_utils.h`
- Modify: `paddle/phi/core/compat/convert_utils.cc`
- Modify: `paddle/phi/core/platform/device_context.h`
- Modify: `paddle/phi/core/platform/device_context.cc`
- Modify: `paddle/phi/backends/all_context.h`
- Modify: `paddle/fluid/framework/data_layout_transform.h`
- Modify: `paddle/fluid/framework/data_layout_transform.cc`
- Modify: host sources that call `phi::funcs` layout-transform helpers through the Fluid header.
- Modify: direct host consumers that name `OneDNNContext` but currently rely on a common aggregate include.

**Interfaces:**
- Consumes: `phi::OneDNNContext` as a complete type only in host oneDNN code.
- Produces: a forward declaration of `phi::OneDNNContext` for generic template declarations and no `dnnl.hpp` include from the three checked public headers.

- [ ] **Step 1: Delete the unused core data-type conversion**

Verify the function has no callers:

```bash
rg -n 'TransToOneDNNDataType' paddle test
```

Expected before deletion: one declaration and one definition only. Remove both guarded blocks and remove `dnnl.hpp` from `convert_utils.h`.

- [ ] **Step 2: Remove oneDNN implementation headers from legacy device context**

Delete the guarded `dnnl.hpp`, layout, and `onednn_context.h` includes from `paddle/phi/core/platform/device_context.h`. Add this host-only include to `paddle/phi/core/platform/device_context.cc`:

```cpp
#ifdef PADDLE_WITH_DNNL
#include "paddle/phi/backends/onednn/onednn_context.h"
#endif
```

- [ ] **Step 3: Replace the aggregate oneDNN include with a forward declaration**

In `paddle/phi/backends/all_context.h`, remove:

```cpp
#include "paddle/phi/backends/onednn/onednn_context.h"
```

and retain the type name required by `kernel_utils.h` without parsing oneDNN:

```cpp
namespace phi {
#ifdef PADDLE_WITH_DNNL
class OneDNNContext;
#endif
}  // namespace phi
```

- [ ] **Step 4: Make complete-type dependencies explicit in host code**

For files that directly name `OneDNNContext`, add a guarded direct include when they do not already include `onednn_context.h`, `onednn_helper.h`, or `onednn_reuse.h`:

```cpp
#ifdef PADDLE_WITH_DNNL
#include "paddle/phi/backends/onednn/onednn_context.h"
#endif
```

Start from this exact inventory and exclude `paddle/phi/core/kernel_utils.h`, which intentionally uses the forward declaration in a template specialization:

```text
paddle/fluid/framework/data_transform.cc
paddle/fluid/framework/operator.cc
paddle/fluid/inference/api/analysis_predictor.cc
paddle/fluid/inference/api/details/zero_copy_tensor.cc
paddle/fluid/operators/controlflow/fetch_op.cc
paddle/fluid/operators/controlflow/fetch_v2_op.cc
paddle/fluid/operators/elementwise/elementwise_op.h
paddle/fluid/operators/fused/fused_matmul_op.cc
paddle/fluid/operators/matmul_op.cc
paddle/fluid/operators/ops_extra_info.h
paddle/phi/core/kernel_registry.cc
paddle/phi/core/platform/device_context.cc
paddle/phi/core/tensor_utils.cc
paddle/phi/kernels/funcs/selected_rows_functor.cc
paddle/phi/kernels/fusion/onednn/fused_matmul_kernel.cc
paddle/phi/kernels/fusion/onednn/fusion_lstm_kernel.cc
paddle/phi/kernels/onednn/conv_function.h
paddle/phi/kernels/onednn/conv_grad_kernel.cc
paddle/phi/kernels/onednn/matmul_grad_kernel.cc
paddle/phi/kernels/onednn/matmul_kernel.cc
test/cpp/fluid/onednn/test_onednn_caching.cc
test/cpp/fluid/onednn/test_onednn_squeeze.cc
```

- [ ] **Step 5: Stop the common Fluid layout-transform header from re-exporting oneDNN**

In `paddle/fluid/framework/data_layout_transform.h`, remove:

```cpp
#include "paddle/phi/backends/onednn/onednn_helper.h"
#include "paddle/phi/kernels/funcs/data_layout_transform.h"
```

Add the backend-neutral declaration dependency explicitly:

```cpp
#include "paddle/common/layout.h"
```

Keep `paddle/phi/kernels/funcs/data_layout_transform.h` in its current path because its direct consumers are host/oneDNN sources. Add it directly to any `.cc` file that calls `TransDataLayoutFromOneDNN`, `GetDataFromTensor`, or `make_memory_desc` and previously received those declarations only through the Fluid header:

```bash
rg -l 'TransDataLayoutFromOneDNN|GetDataFromTensor|make_memory_desc' paddle |
  xargs rg --files-without-match \
    'paddle/phi/kernels/funcs/data_layout_transform\.h'
```

Expected: after adding direct includes, the command returns only the declaration header itself or files whose direct oneDNN backend header already provides the declaration.

- [ ] **Step 6: Re-run the public-header isolation check**

Repeat Task 1 Step 1.

Expected after implementation: preprocessing succeeds for all three headers, each prints `PASS`, and none of the three traces contains `dnnl.hpp`.

- [ ] **Step 7: Format and commit the common-header cleanup**

```bash
pre-commit run clang-format --files $(git diff --name-only --diff-filter=ACM | rg '\.(cc|h)$')
git diff --check
git add \
  paddle/phi/core/compat/convert_utils.h \
  paddle/phi/core/compat/convert_utils.cc \
  paddle/phi/core/platform/device_context.h \
  paddle/phi/core/platform/device_context.cc \
  paddle/phi/backends/all_context.h \
  paddle/fluid/framework/data_layout_transform.h \
  paddle/fluid/framework/data_layout_transform.cc
git diff --name-only --diff-filter=ACM > /tmp/onednn-common-header-files
git add --pathspec-from-file=/tmp/onednn-common-header-files
git commit -m "refactor: remove oneDNN from common headers"
```

Before committing, inspect `git diff --cached --name-only` and keep only Task 4 files.

---

### Task 5: Verify behavior and the CUDA-facing boundary

**Files:**
- Verify: all files changed in Tasks 2-4.
- Verify: existing build directory if configured with `PADDLE_WITH_DNNL=ON`.

**Interfaces:**
- Consumes: the completed backend boundary and migrated call sites.
- Produces: focused local evidence plus the exact remaining CI validation requirement.

- [ ] **Step 1: Run static residue checks**

```bash
rg -n '(->|\.)mem_desc\(|set_mem_desc\(' paddle test
rg -n '#include "dnnl\.hpp"' \
  paddle/phi/core/storage_properties.h \
  paddle/phi/core/dense_tensor.h \
  paddle/phi/core/dense_tensor.inl \
  paddle/phi/core/platform/device_context.h \
  paddle/phi/backends/all_context.h \
  paddle/phi/core/compat/convert_utils.h
git diff --check
```

Expected: both `rg` commands return no matches and `git diff --check` succeeds.

- [ ] **Step 2: Re-run the Task 1 preprocessor check**

Expected: all three public headers preprocess and none reaches `dnnl.hpp`.

- [ ] **Step 3: Run focused formatting and repository hooks**

```bash
pre-commit run clang-format --files $(git diff origin/develop...HEAD --name-only --diff-filter=ACM | rg '\.(cc|h|inl)$')
pre-commit run --files $(git diff origin/develop...HEAD --name-only --diff-filter=ACM)
```

Expected: all applicable hooks pass. Report missing local hook/tooling separately from code failures.

- [ ] **Step 4: Build the focused PHI core and oneDNN targets**

Use the existing configured build directory if one exists. Otherwise configure the smallest available CPU oneDNN build according to the repository `paddle-build` skill, then build the targets owning:

```text
paddle/phi/core/storage_properties.cc
paddle/phi/core/dense_tensor.cc
paddle/phi/core/utils/type_info.cc
paddle/phi/backends/onednn/onednn_context.cc
paddle/phi/kernels/funcs/data_layout_transform.cc
test/cpp/phi/core/test_dense_tensor.cc
```

Expected: the changed core and oneDNN host sources compile with `PADDLE_WITH_DNNL=ON`, and the DenseTensor test target links.

- [ ] **Step 5: Run focused tests available in the configured build**

```bash
PADDLE_BUILD_DIR="${PADDLE_BUILD_DIR:-build}"
ctest --test-dir "$PADDLE_BUILD_DIR" --output-on-failure \
  -R 'test_dense_tensor|onednn'
```

Expected: selected existing tests pass. Record the exact regex, test count, and any unrelated pre-existing failures.

- [ ] **Step 6: Inspect the final branch and commit any verification-only fixes**

```bash
git status --short --branch
git diff origin/develop...HEAD --stat
git log --oneline origin/develop..HEAD
```

The branch must contain only the design, plan, and implementation commits for this isolation fix. Do not include `stash@{0}`.

- [ ] **Step 7: State the external validation boundary**

Local macOS preprocessing and host compilation cannot prove the CUDA 12.9.41/GCC 13.3 `cudafe++` ICE is gone. The final required evidence is a rerun of the Coverage CI job with CUDA C++20; the expected change is that representative `.cu` commands no longer list `dnnl.hpp` in their include path and no longer crash in `std::destroy_at<dnnl_exec_arg_t>`.
