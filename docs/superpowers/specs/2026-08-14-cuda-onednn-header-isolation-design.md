# CUDA oneDNN Header Isolation Design

## Goal

Prevent CUDA translation units from parsing oneDNN C++ headers when Paddle is
built with `PADDLE_WITH_DNNL` and CUDA C++20. Keep oneDNN itself unchanged and
avoid an NVCC-version-specific specialization in `std`.

## Evidence

`paddle/phi/core/dense_tensor.h` includes `storage_properties.h`, which includes
`dnnl.hpp` under `PADDLE_WITH_DNNL`. `dense_tensor.inl` also exposes
`dnnl::memory::desc` through `DenseTensor::mem_desc()` and `set_mem_desc()`.
Consequently, ordinary CUDA kernels that only need `DenseTensor` still make
NVCC parse oneDNN's inline C++ wrappers and trigger the CUDA 12.9/GCC 13 C++20
`std::destroy_at<dnnl_exec_arg_t>` compiler crash.

## Design

1. Keep `StorageProperties` and `DenseTensor` backend-neutral.
2. Move `OneDNNStorageProperties` into `paddle/phi/backends/onednn`.
3. Replace the oneDNN-specific `DenseTensor` member functions with backend-local
   free functions for reading and writing the oneDNN memory descriptor.
4. Update existing oneDNN call sites mechanically; do not change descriptor
   ownership, copying, layout updates, or error behavior.
5. Move remaining oneDNN declarations out of CUDA-reachable common headers,
   including context aggregation, conversion helpers, and layout-transform
   helpers. Add explicit oneDNN includes only to host C++ consumers that use
   those declarations.

The internal API change is intentional: code using `DenseTensor::mem_desc()` or
`set_mem_desc()` must use the backend-local helpers. This is necessary because
the nested type `dnnl::memory::desc` cannot be forward-declared without defining
`dnnl::memory`, so the existing member signatures inherently require
`dnnl.hpp` in the core Tensor header.

## Non-goals

- Do not update or patch oneDNN.
- Do not add a global NVCC forced-include workaround.
- Do not change DenseTensor layout or oneDNN descriptor semantics.
- Do not refactor unrelated backend properties.

## Validation

1. A preprocessor check with `PADDLE_WITH_DNNL` must show that including
   `dense_tensor.h`, `device_context.h`, and `all_context.h` no longer reaches
   `dnnl.hpp`.
2. Compile the changed PHI core and oneDNN host targets with
   `PADDLE_WITH_DNNL`.
3. Compile representative CUDA targets that previously failed under C++20.
4. Run the existing DenseTensor storage-property tests and relevant oneDNN
   tests.
5. Re-run the CUDA 12.9/GCC 13 Coverage CI job; local macOS checks cannot prove
   that exact NVCC failure is gone.
