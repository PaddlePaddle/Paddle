#pragma once

// HIP-clang compiles many translation units with `-x hip` but without defining
// the CUDA preprocessor tokens that CCCL's `nv/target` + `nv/detail/__target_macros`
// use to select the real NVCC/clang-cuda implementation.
//
// When those tokens are missing, CCCL falls back to a "host-only NVCC" stub which
// breaks `NV_DISPATCH_TARGET(...)` / `NV_IF_TARGET(...)` expansions and produces
// parse errors inside headers like `libcudacxx/include/cuda/std/.../__cuda/chrono.h`.
//
// Fix: advertise a CUDA-like compilation mode *just enough* for CCCL's target
// machinery to pick the NVCC/clang-cuda branch. This is intentionally narrow:
// only enabled for HIP translation units (`__HIP__`).
//
// Do **not** define `__CUDACC__` here: HIP-clang may already define it in some
// passes, and pretending to be "CUDA compiling" breaks unrelated headers that
// key off `__CUDACC__` before `__HIPCC__` (e.g. Paddle's bf16 shim includes
// `<cuda_bf16.h>`).
//
// Do **not** define fake `__NVCC__` on HIP either: it makes Paddle headers
// (e.g. `cub.h`) pull CUDA-only includes. CCCL `nv/target` is patched to treat
// `__HIPCC__` as the NVCC dispatch path (see `third_party/cccl/.../nv/target`).
