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

import time

import numpy as np
import torch

import paddle

# import nvtx

cuda_device_num = 0

torch_type = {"float32": torch.float32, "float16": torch.float16}
paddle_type = {"float32": paddle.float32, "float16": paddle.float16}


def convert_numpy(frame_name, data, cuda_device_num=0):
    if isinstance(data, np.ndarray):
        if frame_name == "paddle":
            return paddle.to_tensor(data).cuda(cuda_device_num)
        elif frame_name == "torch":
            return torch.tensor(data).cuda(cuda_device_num)
        else:
            raise NotImplementedError
    elif isinstance(data, list):
        return [convert_numpy(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_numpy(item) for item in data)
    else:
        return data


def set_item_bench(
    n,
    index,
    n_repeat,
    n_warmup,
    score_list,
    cuda_device_num=0,
    dtype="float32",
    is_tensor=False,
):

    x = paddle.to_tensor(
        n, dtype=paddle_type[dtype], place=paddle.CUDAPlace(cuda_device_num)
    )
    y = torch.tensor(
        n, dtype=torch_type[dtype], device=f"cuda:{cuda_device_num}"
    )

    paddle.device.synchronize()
    start_event = [
        paddle.device.Event(enable_timing=True) for i in range(n_repeat)
    ]
    end_event = [
        paddle.device.Event(enable_timing=True) for i in range(n_repeat)
    ]
    cpu_exec_times = 0

    if isinstance(index, np.ndarray):
        index_p = convert_numpy("paddle", index, cuda_device_num=0)
    else:
        index_p = index

    if is_tensor:
        x_value = paddle.full(x[index_p].shape, 0.5, paddle_type[dtype])
    else:
        x_value = 0.5

    paddle.device.synchronize()
    # warmup
    for _ in range(n_warmup):
        x[index_p] = x_value
    paddle.device.synchronize()
    for i in range(n_repeat):
        cpu_start = time.perf_counter_ns()
        start_event[i].record()
        x[index_p] = x_value
        end_event[i].record()
        paddle.device.synchronize()
        cpu_end = time.perf_counter_ns()
        cpu_exec_times = cpu_exec_times + (float)((cpu_end - cpu_start) / 1000)
    paddle.device.synchronize()

    gpu_exec_times = paddle.to_tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=paddle.float64,
    )
    cpu_exec_times = paddle.to_tensor(cpu_exec_times, dtype=paddle.float64)

    paddle_gpu = (paddle.mean(gpu_exec_times) * 1000).cpu().numpy().item()
    paddle_cpu = (cpu_exec_times / n_repeat).cpu().numpy().item()

    start_event = [
        torch.cuda.Event(enable_timing=True) for i in range(n_repeat)
    ]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    cpu_exec_times = 0

    if isinstance(index, np.ndarray):
        index_t = convert_numpy("torch", index)
    else:
        index_t = index

    if is_tensor:
        y_value = torch.full(
            y[index_t].shape, 0.5, device=f"cuda:{cuda_device_num}"
        )
    else:
        y_value = 0.5

    torch.cuda.synchronize()
    # warmup
    for _ in range(n_warmup):
        y[index_t] = y_value
    torch.cuda.synchronize()

    for i in range(n_repeat):
        start_event[i].record()
        cpu_start = time.perf_counter_ns()
        y[index_t] = y_value
        end_event[i].record()
        torch.cuda.synchronize()
        cpu_end = time.perf_counter_ns()
        cpu_exec_times = cpu_exec_times + (float)((cpu_end - cpu_start) / 1000)
    torch.cuda.synchronize()

    gpu_exec_times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float64,
    )
    cpu_exec_times = torch.tensor(cpu_exec_times, dtype=torch.float64)

    torch_gpu = (torch.mean(gpu_exec_times) * 1000).cpu().numpy().item()
    torch_cpu = (cpu_exec_times / n_repeat).cpu().numpy().item()
    np_x = x.cpu().numpy()
    np_y = y.cpu().numpy()

    np.testing.assert_allclose(np_x, np_y)
    print(
        f"set_item (scalar) paddle_gpu: {paddle_gpu:.2f} us torch_gpu: {torch_gpu:.2f} us P/T GPU score: {paddle_gpu / torch_gpu:.2f}) "
    )
    print(
        f"set_item (scalar) paddle_cpu: {paddle_cpu:.2f} us torch_cpu: {torch_cpu:.2f} us P/T CPU score: {paddle_cpu / torch_cpu:.2f}) "
    )
    score_list.append(paddle_cpu / torch_cpu)


def get_item_bench(
    n, index, n_repeat, n_warmup, score_list, cuda_device_num=0, dtype="float32"
):
    x = paddle.to_tensor(
        n, dtype=paddle_type[dtype], place=paddle.CUDAPlace(cuda_device_num)
    )
    y = torch.tensor(
        n, dtype=torch_type[dtype], device=f"cuda:{cuda_device_num}"
    )
    paddle_z = paddle.to_tensor(
        n, dtype=paddle_type[dtype], place=paddle.CUDAPlace(cuda_device_num)
    )
    torch_z = torch.tensor(
        n, dtype=torch_type[dtype], device=f"cuda:{cuda_device_num}"
    )

    paddle.device.synchronize()
    start_event = [
        paddle.device.Event(enable_timing=True) for i in range(n_repeat)
    ]
    end_event = [
        paddle.device.Event(enable_timing=True) for i in range(n_repeat)
    ]
    cpu_exec_times = 0

    if isinstance(index, np.ndarray):
        index_p = convert_numpy("paddle", index)
    else:
        index_p = index
    paddle.device.synchronize()
    # warmup
    for _ in range(n_warmup):
        paddle_z = x[index_p]
    paddle.device.synchronize()
    print("paddle out shape ", paddle_z.shape)
    for i in range(n_repeat):
        cpu_start = time.perf_counter_ns()
        start_event[i].record()
        paddle_z = x[index_p]
        end_event[i].record()
        paddle.device.synchronize()
        cpu_end = time.perf_counter_ns()
        cpu_exec_times = cpu_exec_times + (float)((cpu_end - cpu_start) / 1000)
    paddle.device.synchronize()

    gpu_exec_times = paddle.to_tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=paddle.float64,
    )
    cpu_exec_times = paddle.to_tensor(cpu_exec_times, dtype=paddle.float64)

    paddle_gpu = (paddle.mean(gpu_exec_times) * 1000).cpu().numpy().item()
    paddle_cpu = (cpu_exec_times / n_repeat).cpu().numpy().item()

    start_event = [
        torch.cuda.Event(enable_timing=True) for i in range(n_repeat)
    ]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    cpu_exec_times = 0

    if isinstance(index, np.ndarray):
        index_t = convert_numpy("torch", index)
    else:
        index_t = index
    torch.cuda.synchronize()
    # warmup
    for _ in range(n_warmup):
        torch_z = y[index_t]
    torch.cuda.synchronize()
    print("torch out shape ", torch_z.shape)
    for i in range(n_repeat):
        start_event[i].record()
        cpu_start = time.perf_counter_ns()
        torch_z = y[index_t]
        end_event[i].record()
        torch.cuda.synchronize()
        cpu_end = time.perf_counter_ns()
        cpu_exec_times = cpu_exec_times + (float)((cpu_end - cpu_start) / 1000)
    torch.cuda.synchronize()

    gpu_exec_times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float64,
    )
    cpu_exec_times = torch.tensor(cpu_exec_times, dtype=torch.float64)

    torch_gpu = (torch.mean(gpu_exec_times) * 1000).cpu().numpy().item()
    torch_cpu = (cpu_exec_times / n_repeat).cpu().numpy().item()

    np_x = x.cpu().numpy()
    np_y = y.cpu().numpy()
    np.testing.assert_allclose(np_x, np_y)

    print(
        f"get_item paddle_gpu: {paddle_gpu:.2f} us, torch_gpu: {torch_gpu:.2f} us, Paddle/Torch GPU score: {paddle_gpu / torch_gpu:.2f}) "
    )
    print(
        f"get_item paddle_cpu: {paddle_cpu:.2f} us, torch_cpu: {torch_cpu:.2f} us, Paddle/Torch CPU score: {paddle_cpu / torch_cpu:.2f}) "
    )
    score_list.append(paddle_cpu / torch_cpu)


def set_item_grad_bench(
    n,
    index,
    n_repeat,
    n_warmup,
    score_list,
    cuda_device_num=0,
    dtype="float32",
    is_tensor=False,
):
    x = paddle.to_tensor(
        n, dtype=paddle_type[dtype], place=paddle.CUDAPlace(cuda_device_num)
    )
    x.stop_gradient = False
    y = torch.tensor(
        n,
        dtype=torch_type[dtype],
        device=f"cuda:{cuda_device_num}",
        requires_grad=True,
    )

    paddle.device.synchronize()
    start_event = [
        paddle.device.Event(enable_timing=True) for i in range(n_repeat)
    ]
    end_event = [
        paddle.device.Event(enable_timing=True) for i in range(n_repeat)
    ]
    cpu_exec_times = 0

    if isinstance(index, np.ndarray):
        index_p = convert_numpy("paddle", index)
    else:
        index_p = index

    if is_tensor:
        x_value = paddle.full(x[index_p].shape, 0.5, paddle_type[dtype])
    else:
        x_value = 0.5

    paddle.device.synchronize()
    # forward
    paddle_z = x * 1
    paddle_z[index_p] = x_value
    # backward
    grad_outputs = paddle.ones_like(paddle_z)
    # warmup
    for _ in range(n_warmup):
        grad_x = paddle.grad(
            [paddle_z], [x], grad_outputs=grad_outputs, allow_unused=True
        )

    paddle.device.synchronize()
    for i in range(n_repeat):
        cpu_start = time.perf_counter_ns()
        start_event[i].record()
        grad_x = paddle.grad(
            [paddle_z], [x], grad_outputs=grad_outputs, allow_unused=True
        )
        end_event[i].record()
        paddle.device.synchronize()
        cpu_end = time.perf_counter_ns()
        cpu_exec_times = cpu_exec_times + (float)((cpu_end - cpu_start) / 1000)
    paddle.device.synchronize()

    gpu_exec_times = paddle.to_tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=paddle.float64,
    )
    cpu_exec_times = paddle.to_tensor(cpu_exec_times, dtype=paddle.float64)
    paddle_gpu = (paddle.mean(gpu_exec_times) * 1000).cpu().numpy().item()
    paddle_cpu = (cpu_exec_times / n_repeat).cpu().numpy().item()

    start_event = [
        torch.cuda.Event(enable_timing=True) for i in range(n_repeat)
    ]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    cpu_exec_times = 0

    if isinstance(index, np.ndarray):
        index_t = convert_numpy("torch", index)
    else:
        index_t = index
    if is_tensor:
        y_value = torch.full(
            y[index_t].shape,
            0.5,
            device=f"cuda:{cuda_device_num}",
        )
        y_value.stop_gradient = False
    else:
        y_value = 0.5

    torch.cuda.synchronize()
    # forward
    torch_z = y * 1
    torch_z[index_t] = y_value
    # backward
    grad_outputs = torch.ones_like(torch_z, device=f"cuda:{cuda_device_num}")
    # warmup
    for _ in range(n_warmup):
        grad_y = torch.autograd.grad(
            [torch_z], [y], grad_outputs=grad_outputs, retain_graph=True
        )

    torch.cuda.synchronize()

    for i in range(n_repeat):
        start_event[i].record()
        cpu_start = time.perf_counter_ns()
        grad_y = torch.autograd.grad(
            [torch_z], [y], grad_outputs=grad_outputs, retain_graph=True
        )
        end_event[i].record()
        torch.cuda.synchronize()
        cpu_end = time.perf_counter_ns()
        cpu_exec_times = cpu_exec_times + (float)((cpu_end - cpu_start) / 1000)
    torch.cuda.synchronize()

    gpu_exec_times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float64,
    )
    cpu_exec_times = torch.tensor(cpu_exec_times, dtype=torch.float64)
    torch_gpu = (torch.mean(gpu_exec_times) * 1000).cpu().numpy().item()
    torch_cpu = (cpu_exec_times / n_repeat).cpu().numpy().item()
    for i in range(len(grad_x)):
        np.testing.assert_allclose(
            grad_x[i].cpu().numpy(), grad_y[i].cpu().numpy()
        )

    print(
        f"set_item_grad paddle_gpu: {paddle_gpu:.2f} us, torch_gpu: {torch_gpu:.2f} us, Paddle/Torch GPU score: {paddle_gpu / torch_gpu:.2f}) "
    )
    print(
        f"set_item_grad paddle_cpu: {paddle_cpu:.2f} us, torch_cpu: {torch_cpu:.2f} us, Paddle/Torch CPU score: {paddle_cpu / torch_cpu:.2f}) "
    )
    score_list.append(paddle_cpu / torch_cpu)


def get_item_grad_bench(
    n, index, n_repeat, n_warmup, score_list, cuda_device_num=0, dtype="float32"
):
    x = paddle.to_tensor(
        n, dtype=paddle_type[dtype], place=paddle.CUDAPlace(cuda_device_num)
    )
    x.stop_gradient = False
    y = torch.tensor(
        n,
        dtype=torch_type[dtype],
        device=f"cuda:{cuda_device_num}",
        requires_grad=True,
    )

    paddle.device.synchronize()
    start_event = [
        paddle.device.Event(enable_timing=True) for i in range(n_repeat)
    ]
    end_event = [
        paddle.device.Event(enable_timing=True) for i in range(n_repeat)
    ]
    cpu_exec_times = 0

    if isinstance(index, np.ndarray):
        index_p = convert_numpy("paddle", index)
    else:
        index_p = index
    paddle.device.synchronize()
    # forward
    paddle_z = x[index_p]
    # backward
    grad_outputs = paddle.ones_like(paddle_z)
    # warmup
    for _ in range(n_warmup):
        grad_x = paddle.grad(
            [paddle_z], [x], grad_outputs=grad_outputs, allow_unused=True
        )
    paddle.device.synchronize()
    for i in range(n_repeat):
        cpu_start = time.perf_counter_ns()
        start_event[i].record()
        grad_x = paddle.grad(
            [paddle_z], [x], grad_outputs=grad_outputs, allow_unused=True
        )
        end_event[i].record()
        paddle.device.synchronize()
        cpu_end = time.perf_counter_ns()
        cpu_exec_times = cpu_exec_times + (float)((cpu_end - cpu_start) / 1000)
    paddle.device.synchronize()

    gpu_exec_times = paddle.to_tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=paddle.float64,
    )
    cpu_exec_times = paddle.to_tensor(cpu_exec_times, dtype=paddle.float64)
    paddle_gpu = (paddle.mean(gpu_exec_times) * 1000).cpu().numpy().item()
    paddle_cpu = (cpu_exec_times / n_repeat).cpu().numpy().item()

    start_event = [
        torch.cuda.Event(enable_timing=True) for i in range(n_repeat)
    ]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    cpu_exec_times = 0

    if isinstance(index, np.ndarray):
        index_t = convert_numpy("torch", index)
    else:
        index_t = index
    torch.cuda.synchronize()
    # forward
    torch_z = y[index_t]
    # backward
    grad_outputs = torch.ones_like(torch_z, device=f"cuda:{cuda_device_num}")
    # warmup
    for _ in range(n_warmup):
        grad_y = torch.autograd.grad(
            [torch_z], [y], grad_outputs=grad_outputs, retain_graph=True
        )
    torch.cuda.synchronize()

    for i in range(n_repeat):
        start_event[i].record()
        cpu_start = time.perf_counter_ns()
        grad_y = torch.autograd.grad(
            [torch_z], [y], grad_outputs=grad_outputs, retain_graph=True
        )
        end_event[i].record()
        torch.cuda.synchronize()
        cpu_end = time.perf_counter_ns()
        cpu_exec_times = cpu_exec_times + (float)((cpu_end - cpu_start) / 1000)
    torch.cuda.synchronize()

    gpu_exec_times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float64,
    )
    cpu_exec_times = torch.tensor(cpu_exec_times, dtype=torch.float64)

    torch_gpu = (torch.mean(gpu_exec_times) * 1000).cpu().numpy().item()
    torch_cpu = (cpu_exec_times / n_repeat).cpu().numpy().item()

    for i in range(len(grad_x)):
        np.testing.assert_allclose(
            grad_x[i].cpu().numpy(), grad_y[i].cpu().numpy()
        )

    print(
        f"get_item_grad paddle_gpu: {paddle_gpu:.2f} us, torch_gpu: {torch_gpu:.2f} us, Paddle/Torch GPU score: {paddle_gpu / torch_gpu:.2f}) "
    )
    print(
        f"get_item_grad paddle_cpu: {paddle_cpu:.2f} us, torch_cpu: {torch_cpu:.2f} us, Paddle/Torch CPU score: {paddle_cpu / torch_cpu:.2f}) "
    )
    score_list.append(paddle_cpu / torch_cpu)


def test_dtype(
    first_index_dict,
    second_index_dict,
    n_repeat,
    n_warmup,
    dtype="float32",
    is_tensor=False,
):
    print("========== test ", dtype, " is_tensor ", is_tensor, "=============")
    get_item_score = []
    get_item_grad_score = []
    set_item_score = []
    set_item_grad_score = []
    for key in first_index_dict:
        index_list = first_index_dict[key]
        print(key, " case :")
        n = np.random.randn(108, 64, 12288).astype(dtype)
        print("x.shape = ", n.shape)
        i = 0
        for index in index_list:
            i += 1
            print("index = ", str(index))
            if not is_tensor:
                get_item_bench(
                    n,
                    index,
                    n_repeat,
                    n_warmup,
                    get_item_score,
                    cuda_device_num=cuda_device_num,
                    dtype=dtype,
                )
                get_item_grad_bench(
                    n,
                    index,
                    n_repeat,
                    n_warmup,
                    get_item_grad_score,
                    cuda_device_num=cuda_device_num,
                    dtype=dtype,
                )

            set_item_bench(
                n,
                index,
                n_repeat,
                n_warmup,
                set_item_score,
                cuda_device_num=cuda_device_num,
                dtype=dtype,
                is_tensor=is_tensor,
            )

            if key == "combined" and i == 3:
                continue
            set_item_grad_bench(
                n,
                index,
                n_repeat,
                n_warmup,
                set_item_grad_score,
                cuda_device_num=cuda_device_num,
                dtype=dtype,
                is_tensor=is_tensor,
            )
            print(" ")

    for key in second_index_dict:
        index_list = second_index_dict[key]
        print(key, " case :")
        n = np.random.randn(108, 64, 12288, 3).astype(dtype)
        print("x.shape = ", n.shape)
        for index in index_list:
            print("index = ", str(index))
            if not is_tensor:
                get_item_bench(
                    n,
                    index,
                    n_repeat,
                    n_warmup,
                    get_item_score,
                    cuda_device_num=cuda_device_num,
                    dtype=dtype,
                )
                get_item_grad_bench(
                    n,
                    index,
                    n_repeat,
                    n_warmup,
                    get_item_grad_score,
                    cuda_device_num=cuda_device_num,
                    dtype=dtype,
                )

            set_item_bench(
                n,
                index,
                n_repeat,
                n_warmup,
                set_item_score,
                cuda_device_num=cuda_device_num,
                dtype=dtype,
                is_tensor=is_tensor,
            )
            set_item_grad_bench(
                n,
                index,
                n_repeat,
                n_warmup,
                set_item_grad_score,
                cuda_device_num=cuda_device_num,
                dtype=dtype,
                is_tensor=is_tensor,
            )
            print(" ")
    score_lists = [
        get_item_score,
        set_item_score,
        get_item_grad_score,
        set_item_grad_score,
    ]
    name_list = [
        "get_item_score",
        "set_item_score",
        "get_item_grad_score",
        "set_item_grad_score",
    ]

    for name, score_list in zip(name_list, score_lists):
        G = 0
        S = 0
        B = 0
        B2 = 0
        Bother = 0
        for item in score_list:
            if item <= 0.90:
                G += 1
            elif item > 0.9 and item <= 1.1:
                S += 1
            else:
                B += 1
                if item <= 2:
                    B2 += 1
                else:
                    Bother += 1
        print(
            name,
            "total_case = ",
            G + S + B,
            ", G = ",
            G,
            ", S = ",
            S,
            ", B = ",
            B,
            ", score <=2 : ",
            B2,
            ", score >2 : ",
            Bother,
        )

    forward_score_list = get_item_score + set_item_score
    backward_score_list = get_item_grad_score + set_item_grad_score

    total_score = 0
    for score in forward_score_list:
        total_score += score
    forward_avg_score = total_score / len(forward_score_list)

    total_score = 0
    for score in backward_score_list:
        total_score += score
    backward_avg_score = total_score / len(backward_score_list)

    print("forward_avg_score = ", forward_avg_score)
    print("backward_avg_score = ", backward_avg_score)


def main():
    cuda_device_num = 0

    paddle.device.set_device(f"gpu:{cuda_device_num}")
    torch.cuda.set_device(f"cuda:{cuda_device_num}")

    first_index_dict = {
        "scalar": [0, (2, 2, -1)],
        "slice": [
            slice(None, None, None),
            slice(0, 100, 2),
            (slice(0, 4, 2), slice(None, None, None), slice(1, -1, None)),
        ],
        "none": [None, (0, 0, 0, None)],
        "ellipsis": [Ellipsis],
        "tuple": [(), (0, slice(None, None, None), 0, None)],
        "bool": [
            True,
            np.ones((108), dtype=bool),
            np.ones((108, 64), dtype=bool),
            np.ones((108, 64, 12288), dtype=bool),
        ],
        "list": [
            [1, 0, 2],
            (
                [
                    0,
                    1,
                ],
                [
                    3,
                    2,
                ],
                [
                    0,
                    2,
                ],
            ),
        ],
        "tensor": [
            # np.ones((10, 10, 10), dtype=np.int64), # out of memory
            np.ones((2, 4, 6), dtype=np.int64),
            np.ones((), dtype=np.int64),
            (np.ones((2), dtype=np.int64), np.ones((2), dtype=np.int64)),
        ],
        "combined": [
            (
                slice(None, None, None),
                3,
                [
                    0,
                    2,
                ],
            ),
            (
                slice(None, None, None),
                [
                    0,
                ],
                None,
                0,
            ),
            (
                slice(None, None, None),
                [
                    0,
                ],
                None,
            ),  # set_item 反向报错
            (
                slice(None, None, None),
                [
                    0,
                ],
                0,
            ),
            (np.ones((108), dtype=bool), slice(None, None, None), -1),
            (slice(0, 4, 2), 3, [0, 2]),
        ],
    }

    second_index_dict = {
        "combined": [
            (
                1,
                [
                    1,
                    2,
                ],
                slice(None, None, None),
                np.ones((2), dtype=np.int64),
            ),
            (slice(None, None, None), [1, 2], slice(None, None, None), 1),
            (
                slice(None, None, None),
                [1, 2],
                slice(None, None, None),
                [1],
            ),  # set_item_grad OOM
        ]
    }

    n_repeat = 80
    n_warmup = 5

    print(" n_repeat = ", n_repeat)
    print(" n_warmup = ", n_warmup)

    test_dtype(
        first_index_dict, second_index_dict, n_repeat, n_warmup, dtype="float32"
    )
    print()
    test_dtype(
        first_index_dict,
        second_index_dict,
        n_repeat,
        n_warmup,
        dtype="float32",
        is_tensor=True,
    )
    print()
    test_dtype(
        first_index_dict, second_index_dict, n_repeat, n_warmup, dtype="float16"
    )


if __name__ == "__main__":
    main()
