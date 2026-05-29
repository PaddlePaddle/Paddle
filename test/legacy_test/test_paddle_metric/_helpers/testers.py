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

import pickle
from collections.abc import Sequence
from copy import deepcopy
from functools import partial
from typing import Any, Callable

import numpy as np
import pytest
from unittests import NUM_PROCESSES, _reference_cachier
from unittests._helpers import _IS_WINDOWS

import paddle
from paddle import Tensor
from paddle.metric import Metric
from paddle.metric.utils.data import _flatten, apply_to_collection


def _sort_if_needed(arr: np.ndarray) -> np.ndarray:
    """Sort a numpy array for comparison.

    - If 1D, returns a sorted copy of the array.
    - If 2D or higher, sorts the rows lexicographically (dictionary order).
    - Otherwise, returns the array unchanged.

    Args:
        arr (np.ndarray): The input numpy array.

    Returns:
        np.ndarray: The sorted array (or unchanged if not 1D or 2D+).

    Examples:
        >>> import numpy as np
        >>> arr = np.array([[2, 1, 3], [1, 2, 3], [1, 1, 2]])
        >>> _sort_if_needed(arr)
        array([[1, 1, 2],
               [1, 2, 3],
               [2, 1, 3]])

    """
    if arr.ndim == 1:
        return np.sort(arr)
    if arr.ndim > 1:
        return arr[np.lexsort(arr.T[::-1])]
    return arr


def _assert_allclose(
    tm_result: Any,
    ref_result: Any,
    atol: float = 1e-08,
    key: str | None = None,
    check_ddp_sorting: bool = False,
) -> None:
    """Recursively assert that two results are within a certain tolerance."""
    if isinstance(tm_result, paddle.Tensor):
        tm_result_np = tm_result.detach().cpu().numpy()
        ref_result_np = (
            ref_result.detach().cpu().numpy()
            if isinstance(ref_result, paddle.Tensor)
            else ref_result
        )
        if check_ddp_sorting:
            if tm_result_np.ndim != ref_result_np.ndim:
                raise ValueError(
                    f"Dimension mismatch: tm_result_np.ndim={tm_result_np.ndim}, ref_result_np.ndim={ref_result_np.ndim}"
                )
            tm_result_np = _sort_if_needed(tm_result_np)
            ref_result_np = _sort_if_needed(ref_result_np)
        np.testing.assert_allclose(
            tm_result_np,
            ref_result_np,
            atol=atol,
            equal_nan=True,
            err_msg=f"tm_result: {tm_result_np}, ref_result: {ref_result_np}",
        )
    elif isinstance(tm_result, Sequence):
        for pl_res, ref_res in zip(tm_result, ref_result):
            _assert_allclose(
                pl_res, ref_res, atol=atol, check_ddp_sorting=check_ddp_sorting
            )
    elif isinstance(tm_result, dict):
        if key is None:
            raise KeyError("Provide Key for Dict based metric results.")
        tm_result_np = (
            tm_result[key].detach().cpu().numpy()
            if isinstance(tm_result[key], paddle.Tensor)
            else tm_result[key]
        )
        ref_result_np = (
            ref_result.detach().cpu().numpy()
            if isinstance(ref_result, paddle.Tensor)
            else ref_result
        )
        if check_ddp_sorting:
            if tm_result_np.ndim != ref_result_np.ndim:
                raise ValueError(
                    f"Dimension mismatch: tm_result_np.ndim={tm_result_np.ndim}, ref_result_np.ndim={ref_result_np.ndim}"
                )
            tm_result_np = _sort_if_needed(tm_result_np)
            ref_result_np = _sort_if_needed(ref_result_np)
        np.testing.assert_allclose(
            tm_result_np,
            ref_result_np,
            atol=atol,
            equal_nan=True,
            err_msg=f"tm_result: {tm_result_np}, ref_result: {ref_result_np}",
        )
    else:
        raise ValueError("Unknown format for comparison")


def _assert_tensor(tm_result: Any, key: str | None = None) -> None:
    """Recursively check that some input only consists of torch tensors."""
    if isinstance(tm_result, Sequence):
        for plr in tm_result:
            _assert_tensor(plr)
    elif isinstance(tm_result, dict):
        if key is None:
            raise KeyError("Provide Key for Dict based metric results.")
        assert isinstance(tm_result[key], paddle.Tensor)
    else:
        assert isinstance(tm_result, paddle.Tensor)


def _assert_requires_grad(
    metric: Metric, tm_result: Any, key: str | None = None
) -> None:
    """Recursively assert that metric output is consistent with the `is_differentiable` attribute."""
    if isinstance(tm_result, Sequence):
        for plr in tm_result:
            _assert_requires_grad(metric, plr, key=key)
    elif isinstance(tm_result, dict):
        if key is None:
            raise KeyError("Provide Key for Dict based metric results.")
        assert metric.is_differentiable == tm_result[key].requires_grad
    else:
        assert metric.is_differentiable == tm_result.requires_grad


def _class_test(
    rank: int,
    world_size: int,
    preds: paddle.Tensor | list | list[dict[str, paddle.Tensor]],
    target: paddle.Tensor | list | list[dict[str, paddle.Tensor]],
    metric_class: Metric,
    reference_metric: Callable,
    dist_sync_on_step: bool,
    metric_args: dict | None = None,
    check_dist_sync_on_step: bool = True,
    check_batch: bool = True,
    atol: float = 1e-08,
    device: str = "cpu",
    fragment_kwargs: bool = False,
    check_scriptable: bool = True,
    check_state_dict: bool = True,
    check_picklable: bool = True,
    check_ddp_sorting: bool = False,
    **kwargs_update: Any,
):
    """Comparison between class metric and reference metric.

    Args:
        rank: rank of current process
        world_size: number of processes
        preds: torch tensor with predictions
        target: torch tensor with targets
        metric_class: metric class that should be tested
        reference_metric: callable function that is used for comparison
        dist_sync_on_step: bool, if true will synchronize metric state across
            processes at each ``forward()``
        metric_args: dict with additional arguments used for class initialization
        check_dist_sync_on_step: bool, if true will check if the metric is also correctly
            calculated per batch and per device (and not just at the end)
        check_batch: bool, if true will check if the metric is also correctly
            calculated across devices for each batch (and not just at the end)
        atol: absolute tolerance used for comparison of results
        device: determine which device to run on, either 'cuda' or 'cpu'
        fragment_kwargs: whether tensors in kwargs should be divided as `preds` and `target` among processes
        check_scriptable: bool indicating if metric should also be tested if it can be scripted
        check_state_dict: bool indicating if metric should be tested that its state_dict by default is empty
        check_picklable: bool indicating if metric should be tested that it can be pickled
        check_ddp_sorting: bool indicating if metric output should be sorted before comparison
        kwargs_update: Additional keyword arguments that will be passed with preds and
            target when running update on the metric.

    """
    assert len(preds) == len(target)
    num_batches = len(preds)
    assert num_batches % world_size == 0, (
        "Number of batches must be divisible by world_size"
    )
    if not metric_args:
        metric_args = {}
    metric = metric_class(dist_sync_on_step=dist_sync_on_step, **metric_args)
    with pytest.raises(RuntimeError):
        metric.is_differentiable = not metric.is_differentiable
    with pytest.raises(RuntimeError):
        metric.higher_is_better = not metric.higher_is_better
    if check_scriptable:
        paddle.jit.to_static(function=metric)
    clone = metric.clone()
    assert clone is not metric, (
        "Clone is not a different object than the metric"
    )
    assert type(clone) == type(metric), (
        "Type of clone did not match metric type"
    )
    metric = metric.to(device)
    preds = apply_to_collection(preds, Tensor, lambda x: x.to(device))
    target = apply_to_collection(target, Tensor, lambda x: x.to(device))
    kwargs_update = {
        k: (v.to(device) if isinstance(v, paddle.Tensor) else v)
        for k, v in kwargs_update.items()
    }
    if check_picklable:
        pickled_metric = pickle.dumps(metric)
        metric = pickle.loads(pickled_metric)
    metric_clone = deepcopy(metric)
    for i in range(rank, num_batches, world_size):
        batch_kwargs_update = {
            k: (v[i] if isinstance(v, paddle.Tensor) else v)
            for k, v in kwargs_update.items()
        }
        batch_result = metric(preds[i], target[i], **batch_kwargs_update)
        if rank == 0 and world_size == 1 and i == 0:
            metric_clone.update(preds[i], target[i], **batch_kwargs_update)
            update_result = metric_clone.compute()
            if isinstance(batch_result, dict):
                for key in batch_result:
                    _assert_allclose(batch_result, update_result[key], key=key)
            else:
                _assert_allclose(batch_result, update_result)
        if metric.dist_sync_on_step and check_dist_sync_on_step and rank == 0:
            if isinstance(preds, paddle.Tensor):
                ddp_preds = paddle.concat(
                    [preds[i + r] for r in range(world_size)]
                ).cpu()
            else:
                ddp_preds = _flatten([preds[i + r] for r in range(world_size)])
            if isinstance(target, paddle.Tensor):
                ddp_target = paddle.concat(
                    [target[i + r] for r in range(world_size)]
                ).cpu()
            else:
                ddp_target = _flatten(
                    [target[i + r] for r in range(world_size)]
                )
            ddp_kwargs_upd = {
                k: (
                    paddle.concat([v[i + r] for r in range(world_size)]).cpu()
                    if isinstance(v, paddle.Tensor)
                    else v
                )
                for k, v in (
                    kwargs_update if fragment_kwargs else batch_kwargs_update
                ).items()
            }
            ref_batch_result = _reference_cachier(reference_metric)(
                ddp_preds, ddp_target, **ddp_kwargs_upd
            )
            if isinstance(batch_result, dict):
                for key in batch_result:
                    _assert_allclose(
                        batch_result,
                        ref_batch_result[key].numpy(),
                        atol=atol,
                        key=key,
                        check_ddp_sorting=check_ddp_sorting,
                    )
            else:
                _assert_allclose(
                    batch_result,
                    ref_batch_result,
                    atol=atol,
                    check_ddp_sorting=check_ddp_sorting,
                )
        elif check_batch and not metric.dist_sync_on_step:
            batch_kwargs_update = {
                k: (v.cpu() if isinstance(v, paddle.Tensor) else v)
                for k, v in (
                    batch_kwargs_update if fragment_kwargs else kwargs_update
                ).items()
            }
            preds_ = (
                preds[i].cpu() if isinstance(preds, paddle.Tensor) else preds[i]
            )
            target_ = (
                target[i].cpu()
                if isinstance(target, paddle.Tensor)
                else target[i]
            )
            ref_batch_result = _reference_cachier(reference_metric)(
                preds_, target_, **batch_kwargs_update
            )
            if isinstance(batch_result, dict):
                for key in batch_result:
                    _assert_allclose(
                        batch_result,
                        ref_batch_result[key].numpy(),
                        atol=atol,
                        key=key,
                        check_ddp_sorting=check_ddp_sorting,
                    )
            else:
                _assert_allclose(
                    batch_result,
                    ref_batch_result,
                    atol=atol,
                    check_ddp_sorting=check_ddp_sorting,
                )
    assert hash(metric), repr(metric)
    if check_state_dict:
        assert metric.state_dict() == {}
    result = metric.compute()
    if isinstance(result, dict):
        for key in result:
            _assert_tensor(result, key=key)
    else:
        _assert_tensor(result)
    if isinstance(preds, paddle.Tensor):
        total_preds = paddle.concat(
            [preds[i] for i in range(num_batches)]
        ).cpu()
    else:
        total_preds = [item for sublist in preds for item in sublist]
    if isinstance(target, paddle.Tensor):
        total_target = paddle.concat(
            [target[i] for i in range(num_batches)]
        ).cpu()
    elif (
        isinstance(target, list)
        and len(target) > 0
        and isinstance(target[0], dict)
    ):
        total_target = {
            k: paddle.concat([t[k] for t in target]) for k in target[0]
        }
    else:
        total_target = [item for sublist in target for item in sublist]
    total_kwargs_update = {
        k: (
            paddle.concat([v[i] for i in range(num_batches)]).cpu()
            if isinstance(v, paddle.Tensor)
            else v
        )
        for k, v in kwargs_update.items()
    }
    ref_result = _reference_cachier(reference_metric)(
        total_preds, total_target, **total_kwargs_update
    )
    if isinstance(ref_result, dict):
        for key in ref_result:
            _assert_allclose(
                result,
                ref_result[key].numpy(),
                atol=atol,
                key=key,
                check_ddp_sorting=check_ddp_sorting,
            )
    else:
        _assert_allclose(
            result, ref_result, atol=atol, check_ddp_sorting=check_ddp_sorting
        )


def _functional_test(
    preds: paddle.Tensor | list,
    target: paddle.Tensor | list | list[dict[str, paddle.Tensor]],
    metric_functional: Callable,
    reference_metric: Callable,
    metric_args: dict | None = None,
    atol: float = 1e-08,
    device: str = "cpu",
    fragment_kwargs: bool = False,
    **kwargs_update: Any,
):
    """Comparison between functional metric and reference metric.

    Args:
        preds: torch tensor with predictions
        target: torch tensor with targets
        metric_functional: metric functional that should be tested
        reference_metric: callable function that is used for comparison
        metric_args: dict with additional arguments used for class initialization
        atol: absolute tolerance used for comparison of results
        device: determine which device to run on, either 'cuda' or 'cpu'
        fragment_kwargs: whether tensors in kwargs should be divided as `preds` and `target` among processes
        kwargs_update: Additional keyword arguments that will be passed with preds and
            target when running update on the metric.

    """
    p_size = preds.shape[0] if isinstance(preds, paddle.Tensor) else len(preds)
    t_size = (
        target.shape[0] if isinstance(target, paddle.Tensor) else len(target)
    )
    assert p_size == t_size, f"different sizes {p_size} and {t_size}"
    num_batches = p_size
    metric_args = metric_args or {}
    metric = partial(metric_functional, **metric_args)
    if isinstance(preds, paddle.Tensor):
        preds = preds.to(device)
    if isinstance(target, paddle.Tensor):
        target = target.to(device)
    elif isinstance(target, list):
        for i, target_dict in enumerate(target):
            if isinstance(target_dict, dict):
                for k in target_dict:
                    if isinstance(target_dict[k], paddle.Tensor):
                        target[i][k] = target_dict[k].to(device)
    kwargs_update = {
        k: (v.to(device) if isinstance(v, paddle.Tensor) else v)
        for k, v in kwargs_update.items()
    }
    for i in range(num_batches // 2):
        extra_kwargs = {
            k: (v[i] if isinstance(v, paddle.Tensor) else v)
            for k, v in kwargs_update.items()
        }
        tm_result = metric(preds[i], target[i], **extra_kwargs)
        extra_kwargs = {
            k: (v.cpu() if isinstance(v, paddle.Tensor) else v)
            for k, v in (
                extra_kwargs if fragment_kwargs else kwargs_update
            ).items()
        }
        ref_result = _reference_cachier(reference_metric)(
            preds[i].cpu() if isinstance(preds, paddle.Tensor) else preds[i],
            target[i].cpu() if isinstance(target, paddle.Tensor) else target[i],
            **extra_kwargs,
        )
        if isinstance(ref_result, dict):
            for key in ref_result:
                _assert_allclose(
                    tm_result, ref_result[key].numpy(), atol=atol, key=key
                )
        else:
            _assert_allclose(tm_result, ref_result, atol=atol)


def _assert_dtype_support(
    metric_module: Metric | None,
    metric_functional: Callable | None,
    preds: paddle.Tensor,
    target: paddle.Tensor | list[dict[str, paddle.Tensor]],
    device: str = "cpu",
    dtype: paddle.dtype = paddle.float16,
    **kwargs_update: Any,
):
    """Test if a metric can be used with half precision tensors.

    Args:
        metric_module: the metric module to test
        metric_functional: the metric functional to test
        preds: torch tensor with predictions
        target: torch tensor with targets
        device: determine device, either "cpu" or "cuda"
        dtype: dtype to run test with
        kwargs_update: Additional keyword arguments that will be passed with preds and
            target when running update on the metric.

    """
    y_hat = (
        preds[0].to(dtype=dtype, device=device)
        if preds[0].is_floating_point()
        else preds[0].to(device)
    )
    y = (
        target[0].to(dtype=dtype, device=device)
        if isinstance(target[0], paddle.Tensor)
        and target[0].is_floating_point()
        else {
            k: (
                target[0][k].to(dtype=dtype, device=device)
                if target[0][k].is_floating_point()
                else target[0][k].to(device)
            )
            for k in target[0]
        }
        if isinstance(target[0], dict)
        else target[0].to(device)
    )
    kwargs_update = {
        k: (
            (v[0].to(dtype=dtype) if v.is_floating_point() else v[0]).to(device)
            if isinstance(v, paddle.Tensor)
            else v
        )
        for k, v in kwargs_update.items()
    }
    if metric_module is not None:
        metric_module = metric_module.to(device)
        _assert_tensor(metric_module(y_hat, y, **kwargs_update))
    if metric_functional is not None:
        _assert_tensor(metric_functional(y_hat, y, **kwargs_update))


def _select_rand_best_device() -> str:
    """Select the best device to run tests on."""
    nb_gpus = paddle.cuda.device_count()
    if nb_gpus:
        return "cuda"
    return "cpu"


class MetricTester:
    """Test class for all metrics.

    Class used for efficiently run a lot of parametrized tests in DDP mode. Makes sure that DDP is only setup once and
    that pool of processes are used for all tests. All tests should subclass from this and implement a new method called
    ``test_metric_name`` where the method ``self.run_metric_test`` is called inside.

    """

    atol: float = 1e-08

    def run_functional_metric_test(
        self,
        preds: paddle.Tensor,
        target: paddle.Tensor,
        metric_functional: Callable,
        reference_metric: Callable,
        metric_args: dict | None = None,
        fragment_kwargs: bool = False,
        **kwargs_update: Any,
    ):
        """Core method that should be used for testing functions. Call this inside testing method.

        Args:
            preds: torch tensor with predictions
            target: torch tensor with targets
            metric_functional: metric class that should be tested
            reference_metric: callable function that is used for comparison
            metric_args: dict with additional arguments used for class initialization
            fragment_kwargs: whether tensors in kwargs should be divided as `preds` and `target` among processes
            kwargs_update: Additional keyword arguments that will be passed with preds and
                target when running update on the metric.

        """
        _functional_test(
            preds=preds,
            target=target,
            metric_functional=metric_functional,
            reference_metric=reference_metric,
            metric_args=metric_args,
            atol=self.atol,
            device=_select_rand_best_device(),
            fragment_kwargs=fragment_kwargs,
            **kwargs_update,
        )

    def run_class_metric_test(
        self,
        ddp: bool,
        preds: paddle.Tensor | list[dict],
        target: paddle.Tensor | list[dict],
        metric_class: Metric,
        reference_metric: Callable,
        dist_sync_on_step: bool = False,
        metric_args: dict | None = None,
        check_dist_sync_on_step: bool = True,
        check_batch: bool = True,
        fragment_kwargs: bool = False,
        check_scriptable: bool = True,
        check_state_dict: bool = True,
        check_picklable: bool = True,
        atol: float | None = None,
        check_ddp_sorting: bool = False,
        **kwargs_update: Any,
    ):
        """Core method that should be used for testing class. Call this inside testing methods.

        Args:
            ddp: bool, if running in ddp mode or not
            preds: torch tensor with predictions
            target: torch tensor with targets
            metric_class: metric class that should be tested
            reference_metric: callable function that is used for comparison
            dist_sync_on_step: bool, if true will synchronize metric state across processes at each ``forward()``
            metric_args: dict with additional arguments used for class initialization
            check_dist_sync_on_step: bool, if true will check if the metric is also correctly
                calculated per batch and per device (and not just at the end)
            check_batch: bool, if true will check if the metric is also correctly
                calculated across devices for each batch (and not just at the end)
            fragment_kwargs: whether tensors in kwargs should be divided as `preds` and `target` among processes
            check_scriptable: bool indicating if metric should also be tested if it can be scripted
            check_state_dict: bool indicating if metric should be tested that its state_dict by default is empty
            check_picklable: bool indicating if metric should be tested that it can be pickled
            atol: absolute tolerance used for comparison of results, if None will use self.atol
            check_ddp_sorting: bool indicating if metric output should be sorted before comparison
            kwargs_update: Additional keyword arguments that will be passed with preds and
                target when running update on the metric.

        """
        common_kwargs = {
            "preds": preds,
            "target": target,
            "metric_class": metric_class,
            "reference_metric": reference_metric,
            "metric_args": metric_args or {},
            "atol": atol or self.atol,
            "device": _select_rand_best_device(),
            "dist_sync_on_step": dist_sync_on_step,
            "check_dist_sync_on_step": check_dist_sync_on_step,
            "check_batch": check_batch,
            "fragment_kwargs": fragment_kwargs,
            "check_scriptable": check_scriptable,
            "check_state_dict": check_state_dict,
            "check_picklable": check_picklable,
            "check_ddp_sorting": check_ddp_sorting,
        }
        if ddp and hasattr(pytest, "pool"):
            if _IS_WINDOWS:
                pytest.skip("DDP not supported on windows")
            pytest.pool.starmap(
                partial(_class_test, **common_kwargs, **kwargs_update),
                [(rank, NUM_PROCESSES) for rank in range(NUM_PROCESSES)],
            )
        else:
            _class_test(rank=0, world_size=1, **common_kwargs, **kwargs_update)

    @staticmethod
    def run_precision_test_cpu(
        preds: paddle.Tensor,
        target: paddle.Tensor,
        metric_module: Metric | None = None,
        metric_functional: Callable | None = None,
        metric_args: dict | None = None,
        dtype: paddle.dtype = paddle.float16,
        **kwargs_update: Any,
    ) -> None:
        """Test if a metric can be used with half precision tensors on cpu.

        Args:
            preds: torch tensor with predictions
            target: torch tensor with targets
            metric_module: the metric module to test
            metric_functional: the metric functional to test
            metric_args: dict with additional arguments used for class initialization
            dtype: dtype to run test with
            kwargs_update: Additional keyword arguments that will be passed with preds and
                target when running update on the metric.

        """
        metric_args = metric_args or {}
        _assert_dtype_support(
            metric_module(**metric_args) if metric_module is not None else None,
            partial(metric_functional, **metric_args)
            if metric_functional is not None
            else None,
            preds,
            target,
            device="cpu",
            dtype=dtype,
            **kwargs_update,
        )

    @staticmethod
    def run_precision_test_gpu(
        preds: paddle.Tensor,
        target: paddle.Tensor,
        metric_module: Metric | None = None,
        metric_functional: Callable | None = None,
        metric_args: dict | None = None,
        dtype: paddle.dtype = paddle.float16,
        **kwargs_update: Any,
    ) -> None:
        """Test if a metric can be used with half precision tensors on gpu.

        Args:
            preds: torch tensor with predictions
            target: torch tensor with targets
            metric_module: the metric module to test
            metric_functional: the metric functional to test
            metric_args: dict with additional arguments used for class initialization
            dtype: dtype to run test with
            kwargs_update: Additional keyword arguments that will be passed with preds and
                target when running update on the metric.

        """
        metric_args = metric_args or {}
        _assert_dtype_support(
            metric_module(**metric_args) if metric_module is not None else None,
            partial(metric_functional, **metric_args)
            if metric_functional is not None
            else None,
            preds,
            target,
            device="cuda",
            dtype=dtype,
            **kwargs_update,
        )

    @staticmethod
    def run_differentiability_test(
        preds: paddle.Tensor,
        target: paddle.Tensor,
        metric_module: Metric,
        metric_functional: Callable | None = None,
        metric_args: dict | None = None,
    ) -> None:
        """Test if a metric is differentiable or not.

        Args:
            preds: torch tensor with predictions
            target: torch tensor with targets
            metric_module: the metric module to test
            metric_functional: functional version of the metric
            metric_args: dict with additional arguments used for class initialization

        """
        metric_args = metric_args or {}
        metric = metric_module(**metric_args)
        if preds.is_floating_point():
            preds.stop_gradient = not True
            out = metric(preds[0, :2], target[0, :2])
            _assert_requires_grad(metric, out)
            if metric.is_differentiable and metric_functional is not None:
                pass  # TODO: implement differentiability test for paddle


class DummyMetric(Metric):
    """DummyMetric for testing core components."""

    name = "Dummy"
    full_state_update: bool | None = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.declare("x", paddle.tensor(0.0), dist_reduce_fn="sum")

    def update(self):
        """Update state."""

    def compute(self):
        """Compute value."""


class DummyListMetric(Metric):
    """DummyListMetric for testing core components."""

    name = "DummyList"
    full_state_update: bool | None = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.declare("x", [], dist_reduce_fn="cat")

    def update(self, x=None):
        """Update state."""
        x = paddle.tensor(1) if x is None else x
        self.x.append(x)

    def compute(self):
        """Compute value."""
        return self.x


class DummyMetricSum(DummyMetric):
    """DummyMetricSum for testing core components."""

    def update(self, x):
        """Update state."""
        self.x += x

    def compute(self):
        """Compute value."""
        return self.x


class DummyMetricDiff(DummyMetric):
    """DummyMetricDiff for testing core components."""

    def update(self, y):
        """Update state."""
        self.x -= y

    def compute(self):
        """Compute value."""
        return self.x


class DummyMetricMultiOutput(DummyMetricSum):
    """DummyMetricMultiOutput for testing core components."""

    def compute(self):
        """Compute value."""
        return [self.x, self.x]


class DummyMetricMultiOutputDict(DummyMetricSum):
    """DummyMetricMultiOutput for testing core components."""

    def compute(self):
        """Compute value."""
        return {"output1": self.x, "output2": self.x}


def inject_ignore_index(x: paddle.Tensor, ignore_index: int) -> paddle.Tensor:
    """Injecting the ignored index value into a tensor randomly."""
    if any(x.flatten() == ignore_index):
        return x
    classes = paddle.unique(x)
    idx = paddle.randperm(x.size)
    x = deepcopy(x)
    skip = paddle.randint(low=9, high=11, shape=(1,)).item()
    x.view(-1)[idx[::skip]] = ignore_index
    for batch in x:
        new_classes = paddle.unique(batch)
        class_not_in = [(c not in new_classes) for c in classes]
        if any(class_not_in):
            missing_class = int(np.where(class_not_in)[0][0])
            batch[paddle.where(batch == ignore_index)[0][0]] = missing_class
    return x


def remove_ignore_index(
    target: paddle.Tensor, preds: paddle.Tensor, ignore_index: int | None
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Remove samples that are equal to the ignore_index in comparison functions.

    Example:
        >>> target = paddle.to_tensor([0, 1, 2, 3, 4])
        >>> preds = paddle.to_tensor([0, 1, 2, 3, 4])
        >>> ignore_index = 2
        >>> remove_ignore_index(target, preds, ignore_index)
        (tensor([0, 1, 3, 4]), tensor([0, 1, 3, 4]))

    """
    if ignore_index is not None:
        idx = target == ignore_index
        target, preds = deepcopy(target[~idx]), deepcopy(preds[~idx])
    return target, preds


def remove_ignore_index_groups(
    target: paddle.Tensor,
    preds: paddle.Tensor,
    groups: paddle.Tensor,
    ignore_index: int | None,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Version of the remove_ignore_index which includes groups."""
    if ignore_index is not None:
        idx = target == ignore_index
        target, preds, groups = (
            deepcopy(target[~idx]),
            deepcopy(preds[~idx]),
            deepcopy(groups[~idx]),
        )
    return target, preds, groups
