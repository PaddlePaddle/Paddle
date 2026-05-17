# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

os.environ.setdefault("FLAGS_minloglevel", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import paddle
from paddle.jit.sot import symbolic_translate
from paddle.jit.sot.opcode_translator.executor.executor_cache import (
    OpcodeExecutorCache,
)
from paddle.jit.sot.utils import (
    ENV_SOT_ALLOW_DYNAMIC_SHAPE,
    ENV_SOT_ENABLE_COMPILED_GUARD,
    ENV_SOT_ENABLE_STRICT_GUARD_CHECK,
    ENV_SOT_UNSAFE_CACHE_FASTPATH,
)
from paddle.utils.environments import EnvironmentVariableGuard

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class BenchResult:
    name: str
    median_ns: float
    samples_ns: list[float]

    @property
    def median_us(self) -> float:
        return self.median_ns / 1000.0


def paddle_guard_case(x, cfg):
    if cfg["enabled"]:
        return x + cfg["biases"][0]
    return x - cfg["biases"][1]


def paddle_guard_helper(x, scale):
    return x + scale


def paddle_complex_guard_case(x, cfg, seq, scale, fn):
    y = x
    if cfg["enabled"]:
        y = y + cfg["biases"][0]
    if seq[0] < seq[1]:
        y = fn(y, scale)
    return y + cfg["biases"][2]


def paddle_resnet18_case(x, net):
    return net(x)


def resnet_image_sizes(base_size: int, count: int) -> list[int]:
    return [base_size + i * 8 for i in range(count)]


def make_paddle_case(case: str, resnet_image_size: int = 64):
    x = paddle.ones([2, 3])
    if case == "basic":
        return paddle_guard_case, (x, {"enabled": True, "biases": [1, 2]})
    if case == "complex":
        return (
            paddle_complex_guard_case,
            (
                x,
                {"enabled": True, "biases": [1, 2, 3]},
                [1, 2, 3],
                4,
                paddle_guard_helper,
            ),
        )
    if case == "resnet18":
        from paddle.vision.models.resnet import resnet18

        x = paddle.rand((1, 3, resnet_image_size, resnet_image_size))
        net = resnet18(pretrained=False)
        net.eval()
        return paddle_resnet18_case, (x, net)
    raise ValueError(f"unknown benchmark case: {case}")


def time_callable(fn: Callable[[], None], iterations: int, rounds: int):
    samples = []
    for _ in range(rounds):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        samples.append((time.perf_counter_ns() - start) / iterations)
    return samples


def bench_paddle(
    name: str,
    enable_compiled_guard: bool,
    case: str,
    iterations: int,
    rounds: int,
    resnet_image_size: int,
) -> BenchResult:
    OpcodeExecutorCache().clear()
    paddle_case, args = make_paddle_case(case, resnet_image_size)
    fn = symbolic_translate(paddle_case)

    with (
        EnvironmentVariableGuard(
            ENV_SOT_ENABLE_COMPILED_GUARD, enable_compiled_guard
        ),
        EnvironmentVariableGuard(ENV_SOT_ENABLE_STRICT_GUARD_CHECK, False),
        EnvironmentVariableGuard(ENV_SOT_UNSAFE_CACHE_FASTPATH, False),
    ):
        fn(*args)

        def call():
            fn(*args)

        samples = time_callable(call, iterations, rounds)

    OpcodeExecutorCache().clear()
    return BenchResult(name, statistics.median(samples), samples)


def bench_paddle_guard_only(
    name: str,
    enable_compiled_guard: bool,
    case: str,
    iterations: int,
    rounds: int,
    resnet_image_size: int,
) -> BenchResult:
    OpcodeExecutorCache().clear()
    paddle_case, args = make_paddle_case(case, resnet_image_size)
    fn = symbolic_translate(paddle_case)

    with (
        EnvironmentVariableGuard(
            ENV_SOT_ENABLE_COMPILED_GUARD, enable_compiled_guard
        ),
        EnvironmentVariableGuard(ENV_SOT_ENABLE_STRICT_GUARD_CHECK, False),
        EnvironmentVariableGuard(ENV_SOT_UNSAFE_CACHE_FASTPATH, False),
        EnvironmentVariableGuard(ENV_SOT_ALLOW_DYNAMIC_SHAPE, False),
    ):
        fn(*args)

        cache = OpcodeExecutorCache().cache
        guarded_fns = next(iter(cache.values()))
        OpcodeExecutorCache().compiled_guard_lookups.clear()
        custom_code, guard_fn = guarded_fns[0]

        samples = []

        def measuring_guard(frame):
            result = guard_fn(frame)
            start = time.perf_counter_ns()
            for _ in range(iterations):
                guard_fn(frame)
            samples.append((time.perf_counter_ns() - start) / iterations)
            return result

        guarded_fns[0] = (custom_code, measuring_guard)
        try:
            for _ in range(rounds):
                fn(*args)
        finally:
            guarded_fns[0] = (custom_code, guard_fn)

    OpcodeExecutorCache().clear()
    return BenchResult(name, statistics.median(samples), samples)


def bench_paddle_compiled_guard_lookup_resnet18(
    name: str,
    iterations: int,
    rounds: int,
    resnet_image_size: int,
    cache_count: int,
) -> BenchResult:
    from paddle.vision.models.resnet import resnet18

    OpcodeExecutorCache().clear()
    net = resnet18(pretrained=False)
    net.eval()
    fn = symbolic_translate(paddle_resnet18_case)
    sizes = resnet_image_sizes(resnet_image_size, cache_count)
    inputs = [paddle.rand((1, 3, size, size)) for size in sizes]

    with (
        EnvironmentVariableGuard(ENV_SOT_ENABLE_COMPILED_GUARD, True),
        EnvironmentVariableGuard(ENV_SOT_ENABLE_STRICT_GUARD_CHECK, False),
        EnvironmentVariableGuard(ENV_SOT_UNSAFE_CACHE_FASTPATH, False),
        EnvironmentVariableGuard(ENV_SOT_ALLOW_DYNAMIC_SHAPE, False),
    ):
        for x in inputs:
            fn(x, net)

        cache = OpcodeExecutorCache().cache
        guarded_fns = next(iter(cache.values()))
        if len(guarded_fns) != cache_count:
            raise RuntimeError(
                f"expected {cache_count} cache entries, got {len(guarded_fns)}"
            )
        lookup = next(
            iter(OpcodeExecutorCache().compiled_guard_lookups.values())
        )
        hit_index = cache_count - 1
        samples = []

        def measuring_cache_lookup(
            frame,
            guarded_fns,
            compile_time_for_code,
            compile_time_total,
            **kwargs,
        ):
            lookup_result = lookup.lookup(frame)
            if lookup_result != hit_index:
                raise AssertionError(
                    f"compiled guard lookup returned {lookup_result}, "
                    f"expected {hit_index}"
                )
            start = time.perf_counter_ns()
            for _ in range(iterations):
                lookup.lookup(frame)
            samples.append((time.perf_counter_ns() - start) / iterations)
            return guarded_fns[lookup_result][0]

        cache_obj = OpcodeExecutorCache()
        cache_obj.lookup = measuring_cache_lookup
        try:
            for _ in range(rounds):
                fn(inputs[hit_index], net)
        finally:
            del cache_obj.lookup

    OpcodeExecutorCache().clear()
    return BenchResult(name, statistics.median(samples), samples)


def make_torch_case(case: str, resnet_image_size: int = 64):
    import torch

    x = torch.ones([2, 3])

    def torch_guard_case(x, cfg):
        if cfg["enabled"]:
            return x + cfg["biases"][0]
        return x - cfg["biases"][1]

    def torch_guard_helper(x, scale):
        return x + scale

    def torch_complex_guard_case(x, cfg, seq, scale, fn):
        y = x
        if cfg["enabled"]:
            y = y + cfg["biases"][0]
        if seq[0] < seq[1]:
            y = fn(y, scale)
        return y + cfg["biases"][2]

    def torch_resnet18_case(x, model):
        return model(x)

    if case == "basic":
        cfg = {"enabled": True, "biases": [1, 2]}
        return torch_guard_case, (x, cfg), {"x": x, "cfg": cfg}
    if case == "complex":
        cfg = {"enabled": True, "biases": [1, 2, 3]}
        seq = [1, 2, 3]
        scale = 4
        return (
            torch_complex_guard_case,
            (x, cfg, seq, scale, torch_guard_helper),
            {
                "x": x,
                "cfg": cfg,
                "seq": seq,
                "scale": scale,
                "fn": torch_guard_helper,
            },
        )
    if case == "resnet18":
        from torchvision import models

        x = torch.ones([1, 3, resnet_image_size, resnet_image_size])
        model = models.resnet18(weights=None).eval()
        return torch_resnet18_case, (x, model), {"x": x, "model": model}
    raise ValueError(f"unknown benchmark case: {case}")


def bench_torch_hot_call(
    case: str, iterations: int, rounds: int, resnet_image_size: int
) -> BenchResult | None:
    try:
        import torch._dynamo as dynamo
    except ImportError:
        return None

    dynamo.reset()
    torch_case, args, _locals = make_torch_case(case, resnet_image_size)

    fn = dynamo.optimize("eager")(torch_case)
    fn(*args)

    def call():
        fn(*args)

    samples = time_callable(call, iterations, rounds)
    dynamo.reset()
    return BenchResult(
        "torch_dynamo_eager", statistics.median(samples), samples
    )


def bench_torch_guard_only(
    case: str, iterations: int, rounds: int, resnet_image_size: int
) -> BenchResult | None:
    try:
        import torch._dynamo as dynamo
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list
    except ImportError:
        return None

    dynamo.reset()
    torch_case, args, locals_dict = make_torch_case(case, resnet_image_size)
    fn = dynamo.optimize("eager")(torch_case)
    fn(*args)
    cache_entries = _debug_get_cache_entry_list(torch_case.__code__)
    guard_manager = cache_entries[0].guard_manager
    assert guard_manager.check(locals_dict)

    def call():
        guard_manager.check(locals_dict)

    samples = time_callable(call, iterations, rounds)
    dynamo.reset()
    return BenchResult(
        "torch_dynamo_guard_only", statistics.median(samples), samples
    )


def bench_torch_guard_lookup_resnet18(
    iterations: int,
    rounds: int,
    resnet_image_size: int,
    cache_count: int,
) -> BenchResult | None:
    try:
        import torch
        import torch._dynamo as dynamo
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list
        from torchvision import models
    except ImportError:
        return None

    dynamo.reset()
    old_dynamic_shapes = dynamo.config.dynamic_shapes
    old_automatic_dynamic_shapes = dynamo.config.automatic_dynamic_shapes
    dynamo.config.dynamic_shapes = False
    dynamo.config.automatic_dynamic_shapes = False

    def torch_resnet18_case(x, model):
        return model(x)

    try:
        model = models.resnet18(weights=None).eval()
        sizes = resnet_image_sizes(resnet_image_size, cache_count)
        inputs = [torch.rand((1, 3, size, size)) for size in sizes]
        fn = dynamo.optimize("eager")(torch_resnet18_case)
        for x in inputs:
            fn(x, model)

        cache_entries = _debug_get_cache_entry_list(
            torch_resnet18_case.__code__
        )
        hit_locals = {"x": inputs[-1], "model": model}
        miss_managers = []
        hit_manager = None
        for entry in cache_entries:
            if entry.guard_manager.check(hit_locals):
                hit_manager = entry.guard_manager
            else:
                miss_managers.append(entry.guard_manager)
        if hit_manager is None or len(miss_managers) < cache_count - 1:
            raise RuntimeError(
                f"cannot build {cache_count - 1} misses plus one hit from "
                f"{len(cache_entries)} torch cache entries"
            )
        ordered_managers = [*miss_managers[: cache_count - 1], hit_manager]

        def call():
            for guard_manager in ordered_managers:
                if guard_manager.check(hit_locals):
                    return
            raise AssertionError("torch guard lookup missed")

        samples = time_callable(call, iterations, rounds)
    finally:
        dynamo.reset()
        dynamo.config.dynamic_shapes = old_dynamic_shapes
        dynamo.config.automatic_dynamic_shapes = old_automatic_dynamic_shapes
    return BenchResult(
        f"torch_dynamo_guard_lookup_{cache_count}_cache",
        statistics.median(samples),
        samples,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--hot-iterations", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument(
        "--case", choices=["basic", "complex", "resnet18"], default="basic"
    )
    parser.add_argument("--resnet-image-size", type=int, default=64)
    parser.add_argument("--multi-cache-count", type=int, default=1)
    parser.add_argument("--compare-torch", action="store_true")
    parser.add_argument(
        "--require-speedup",
        type=float,
        default=0.0,
        help=(
            "Fail if python_guard_us / compiled_guard_us is below this value. "
            "Use 0 to report only."
        ),
    )
    parser.add_argument(
        "--max-torch-guard-ratio",
        type=float,
        default=0.0,
        help=(
            "Fail if compiled_guard_us / torch_guard_us is above this value. "
            "Requires --compare-torch. Use 0 to report only."
        ),
    )
    parser.add_argument(
        "--max-torch-multi-lookup-ratio",
        type=float,
        default=0.0,
        help=(
            "Fail if compiled guard multi-cache lookup / torch multi-cache "
            "lookup is above this value. Requires --compare-torch and "
            "--multi-cache-count > 1. Use 0 to report only."
        ),
    )
    args = parser.parse_args()
    hot_iterations = (
        args.iterations if args.hot_iterations is None else args.hot_iterations
    )

    python_guard_only = bench_paddle_guard_only(
        "paddle_python_guard_only",
        False,
        args.case,
        args.iterations,
        args.rounds,
        args.resnet_image_size,
    )
    compiled_guard_only = bench_paddle_guard_only(
        "paddle_compiled_guard_only",
        True,
        args.case,
        args.iterations,
        args.rounds,
        args.resnet_image_size,
    )
    guard_only_speedup = (
        python_guard_only.median_ns / compiled_guard_only.median_ns
    )

    python_hot_call = bench_paddle(
        "paddle_python_hot_call",
        False,
        args.case,
        hot_iterations,
        args.rounds,
        args.resnet_image_size,
    )
    compiled_hot_call = bench_paddle(
        "paddle_compiled_hot_call",
        True,
        args.case,
        hot_iterations,
        args.rounds,
        args.resnet_image_size,
    )
    hot_call_speedup = python_hot_call.median_ns / compiled_hot_call.median_ns

    print(
        f"{python_guard_only.name}: {python_guard_only.median_us:.3f} us/check"
    )
    print(
        f"{compiled_guard_only.name}: "
        f"{compiled_guard_only.median_us:.3f} us/check"
    )
    print(f"compiled_guard_only_speedup: {guard_only_speedup:.2f}x")
    print(f"{python_hot_call.name}: {python_hot_call.median_us:.3f} us/call")
    print(
        f"{compiled_hot_call.name}: {compiled_hot_call.median_us:.3f} us/call"
    )
    print(f"compiled_hot_call_speedup: {hot_call_speedup:.2f}x")

    if args.compare_torch:
        torch_guard = bench_torch_guard_only(
            args.case, args.iterations, args.rounds, args.resnet_image_size
        )
        torch_hot_call = bench_torch_hot_call(
            args.case, hot_iterations, args.rounds, args.resnet_image_size
        )
        if torch_guard is None or torch_hot_call is None:
            print("torch_dynamo_eager: unavailable")
        else:
            guard_ratio = compiled_guard_only.median_ns / torch_guard.median_ns
            hot_call_ratio = (
                compiled_hot_call.median_ns / torch_hot_call.median_ns
            )
            print(f"{torch_guard.name}: {torch_guard.median_us:.3f} us/check")
            print(f"compiled_guard_vs_torch_guard_ratio: {guard_ratio:.2f}x")
            print(
                f"{torch_hot_call.name}: {torch_hot_call.median_us:.3f} us/call"
            )
            print(
                f"compiled_hot_call_vs_torch_hot_call_ratio: "
                f"{hot_call_ratio:.2f}x"
            )
            if (
                args.max_torch_guard_ratio > 0
                and guard_ratio > args.max_torch_guard_ratio
            ):
                raise SystemExit(
                    f"compiled guard / torch guard ratio "
                    f"{guard_ratio:.2f}x is above "
                    f"{args.max_torch_guard_ratio:.2f}x"
                )
        if args.max_torch_guard_ratio > 0 and (
            torch_guard is None or torch_hot_call is None
        ):
            raise SystemExit("torch comparison is unavailable")

    if args.multi_cache_count > 1:
        if args.case != "resnet18":
            raise SystemExit("--multi-cache-count currently supports resnet18")
        paddle_multi_lookup = bench_paddle_compiled_guard_lookup_resnet18(
            f"paddle_compiled_guard_lookup_{args.multi_cache_count}_cache",
            args.iterations,
            args.rounds,
            args.resnet_image_size,
            args.multi_cache_count,
        )
        print(
            f"{paddle_multi_lookup.name}: "
            f"{paddle_multi_lookup.median_us:.3f} us/lookup"
        )
        if args.compare_torch:
            torch_multi_lookup = bench_torch_guard_lookup_resnet18(
                args.iterations,
                args.rounds,
                args.resnet_image_size,
                args.multi_cache_count,
            )
            if torch_multi_lookup is None:
                print("torch_dynamo_guard_lookup_multi_cache: unavailable")
            else:
                multi_lookup_ratio = (
                    paddle_multi_lookup.median_ns / torch_multi_lookup.median_ns
                )
                print(
                    f"{torch_multi_lookup.name}: "
                    f"{torch_multi_lookup.median_us:.3f} us/lookup"
                )
                print(
                    "compiled_guard_lookup_vs_torch_multi_guard_ratio: "
                    f"{multi_lookup_ratio:.2f}x"
                )
                if (
                    args.max_torch_multi_lookup_ratio > 0
                    and multi_lookup_ratio > args.max_torch_multi_lookup_ratio
                ):
                    raise SystemExit(
                        "compiled guard lookup / torch multi-cache lookup ratio "
                        f"{multi_lookup_ratio:.2f}x is above "
                        f"{args.max_torch_multi_lookup_ratio:.2f}x"
                    )
            if (
                args.max_torch_multi_lookup_ratio > 0
                and torch_multi_lookup is None
            ):
                raise SystemExit("torch multi-cache comparison is unavailable")

    if args.require_speedup > 0 and guard_only_speedup < args.require_speedup:
        raise SystemExit(
            f"compiled guard speedup {guard_only_speedup:.2f}x is below "
            f"{args.require_speedup:.2f}x"
        )


if __name__ == "__main__":
    main()
