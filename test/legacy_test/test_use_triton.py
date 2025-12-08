# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import platform
import unittest

import numpy as np

import paddle

if (
    paddle.device.is_compiled_with_cuda()
    and not paddle.device.is_compiled_with_rocm()
    and platform.system().lower() == 'linux'
):
    # isort: off
    from paddle.compat import paddle_triton as triton
    from triton import language as tl
    # isort: on


def do_bench(kernel_call, quantiles, use_cuda_graph=False):
    return triton.testing.do_bench(
        kernel_call, quantiles=quantiles, warmup=1, rep=1
    )


class TestPaddleUseTriton(unittest.TestCase):
    def test_kwargs_without_cuda_graph(self, device: str = 'cuda:0'):
        self._test_kwargs(False, device)

    def _test_kwargs(self, use_cuda_graph: bool, device: str):
        if (
            not paddle.is_compiled_with_cuda()
            or paddle.device.is_compiled_with_rocm()
        ):
            print("Skip Triton tests because no CUDA available.")
            return
        if platform.system().lower() == "windows":
            return
        if use_cuda_graph and not paddle.cuda.is_available():
            print("Skip cuda graph tests because no CUDA available.")
            return

        M, N = 1024, 16
        src = paddle.randn(M * N, device=device)
        dst = paddle.empty(M * N, device=device)

        configs = [
            triton.Config(kwargs={'BLOCK_SIZE_M': 32}),
            triton.Config(kwargs={'BLOCK_SIZE_M': 128}),
        ]

        @triton.autotune(
            configs=configs,
            key=["M"],
            do_bench=lambda kernel, quantiles: do_bench(
                kernel, quantiles, use_cuda_graph
            ),
        )
        @triton.jit
        def _kernel(
            dst,
            src,
            stride_m: tl.constexpr,
            M,
            BLOCK_SIZE_N: tl.constexpr,
            BLOCK_SIZE_M: tl.constexpr,
        ):
            offsets_m = tl.program_id(0) * stride_m + tl.arange(0, BLOCK_SIZE_M)
            offsets_n = tl.arange(0, BLOCK_SIZE_N)
            x = tl.load(
                src + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :]
            )
            tl.store(
                dst + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :], x
            )

        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_M']),)
        _kernel[grid](dst, src, N, M, N)
        # the key word args could be in arbitrary order.
        _kernel[grid](dst=dst, src=src, M=M // 2, stride_m=N, BLOCK_SIZE_N=N)
        assert len(_kernel.cache) == 2

    def test_no_do_bench(self, device: str = 'cuda:0'):
        if (
            not paddle.is_compiled_with_cuda()
            or paddle.device.is_compiled_with_rocm()
        ):
            print("Skip Triton tests because no CUDA available.")
            return
        if platform.system().lower() == "windows":
            return
        M, N = 1024, 16
        src = paddle.randn(M * N, device=device)
        dst = paddle.empty(M * N, device=device)

        configs = [
            triton.Config(kwargs={'BLOCK_SIZE_M': 32}),
            triton.Config(kwargs={'BLOCK_SIZE_M': 128}),
        ]

        @triton.autotune(configs=configs, key=["M"])
        @triton.jit
        def _kernel(
            dst,
            src,
            stride_m: tl.constexpr,
            M,
            BLOCK_SIZE_N: tl.constexpr,
            BLOCK_SIZE_M: tl.constexpr,
        ):
            offsets_m = tl.program_id(0) * stride_m + tl.arange(0, BLOCK_SIZE_M)
            offsets_n = tl.arange(0, BLOCK_SIZE_N)
            x = tl.load(
                src + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :]
            )
            tl.store(
                dst + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :], x
            )

        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_M']),)
        _kernel[grid](dst, src, N, M, N)
        assert len(_kernel.cache) == 1

    def test_restore_without_kwargs(self, device='cuda:0'):
        self._test_restore(False, device)

    def _test_restore(self, pass_kwargs_to_kernel, device):
        if (
            not paddle.is_compiled_with_cuda()
            or paddle.device.is_compiled_with_rocm()
        ):
            print("Skip Triton tests because no CUDA available.")
            return
        if platform.system().lower() == "windows":
            return
        N = 1024
        src = paddle.zeros(N, device=device)

        configs = [
            triton.Config(kwargs={'BLOCK_SIZE': 32}),
            triton.Config(kwargs={'BLOCK_SIZE': 128}),
        ]

        @triton.autotune(
            configs=configs, key=['N'], restore_value=['src'], do_bench=do_bench
        )
        @triton.jit
        def _kernel(src, N, BLOCK_SIZE: tl.constexpr):
            offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            x = tl.load(src + offsets, mask=offsets < N) + 1
            tl.store(src + offsets, x, mask=offsets < N)

        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
        if pass_kwargs_to_kernel:
            _kernel[grid](src=src, N=N)
        else:
            _kernel[grid](src, N)
        triton.testing.assert_close(src, paddle.ones_like(src))

    def test_hooks(self, device='cuda:0'):
        if (
            not paddle.is_compiled_with_cuda()
            or paddle.device.is_compiled_with_rocm()
        ):
            print("Skip Triton tests because no CUDA available.")
            return
        if platform.system().lower() == "windows":
            return
        # Autotuner's pre- and post- hooks should be called the same number of times
        N = 4096
        src = paddle.zeros(N, device=device)

        configs = [
            triton.Config(kwargs={'BLOCK_SIZE': 4096}),
            triton.Config(kwargs={'BLOCK_SIZE': 32}),
        ]

        values = {"counter": 0, "has_exception": False}

        def _pre_hook(*args, **kwargs):
            values["counter"] += 1

        def _post_hook(*args, exception):
            values["counter"] -= 1
            if exception is not None:
                values["has_exception"] = True
            assert values["counter"] == 0

        @triton.autotune(
            configs=configs,
            key=['N'],
            do_bench=do_bench,
            pre_hook=_pre_hook,
            post_hook=_post_hook,
        )
        @triton.heuristics(
            {"N_STAGES": lambda nargs: 100 if nargs['N'] == 4096 else 4}
        )
        @triton.jit
        def _kernel(src, N, N_STAGES: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            offsets = tl.arange(0, BLOCK_SIZE)
            max_iters = tl.cdiv(N, BLOCK_SIZE)
            for _ in tl.range(max_iters, num_stages=N_STAGES):
                x = tl.load(src + offsets, mask=offsets < N)
                tl.store(src + offsets, x, mask=offsets < N)
                offsets += BLOCK_SIZE

        _kernel[(1,)](src, N)

        # On NVIDIA GPUs:
        # The tuning knob `num_stages` can be set by users.
        # This will cause out of resources when N_STAGES = 100
        # shared memory bytes = N_STAGES * BLOCK_SIZE * sizeof(float)
        # On AMD GPUs:
        # `num_stages` is a fixed value of 2, so it won't cause out of resources

    def test_prune_configs_without_perf_model(self, device: str = 'cuda:0'):
        self._test_prune_configs(False, device)

    def _test_prune_configs(self, with_perf_model: bool, device: str):
        if (
            not paddle.is_compiled_with_cuda()
            or paddle.device.is_compiled_with_rocm()
        ):
            print("Skip Triton tests because no CUDA available.")
            return
        if platform.system().lower() == "windows":
            return
        N = 1024
        src = paddle.randn(N, device=device)
        dst = paddle.empty(N, device=device)
        records = {}

        def early_config_prune(configs, named_args, **kwargs):
            records['run_early_config_prune'] = True
            if "N" in kwargs and kwargs["N"] == 1024:
                records['capture_kwargs'] = True
            if (
                "dst" in named_args
                and "src" in named_args
                and len(named_args) == 2
            ):
                records['capture_named_args'] = True
            return [configs[0]]

        def perf_model(*args, **kwargs):
            records['run_perf_model'] = True
            return kwargs['BLOCK_SIZE']

        configs = [
            triton.Config(kwargs={'BLOCK_SIZE': 32}),
            triton.Config(kwargs={'BLOCK_SIZE': 128}),
        ]

        if with_perf_model:
            prune_configs_by = {'perf_model': perf_model, 'top_k': 1}
        else:
            prune_configs_by = {'early_config_prune': early_config_prune}

        @triton.autotune(
            configs=configs,
            key=['N'],
            prune_configs_by=prune_configs_by,
            do_bench=do_bench,
        )
        @triton.jit
        def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
            offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            x = tl.load(src + offsets, mask=offsets < N)
            tl.store(dst + offsets, x, mask=offsets < N)

        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
        _kernel[grid](dst, src, N=N)
        np.testing.assert_allclose(src.cpu(), dst.cpu())


if __name__ == '__main__':
    unittest.main()
