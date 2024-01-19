# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import logging

import numpy as np
from dist_amp_base import (
    MLP,
    RandomDataset,
    compare_state_dict,
    create_optimizer,
    save_model_parameters,
)

import paddle
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients,
)

logging.basicConfig(level="INFO", format="%(message)s")


def train_mlp(
    model, train_loader, use_pure_bf16=False, use_main_grad=False, acc_steps=1
):
    logging.info(
        f"-- Train Info: use_pure_bf16={use_pure_bf16}, use_main_grad={use_main_grad}, acc_steps={acc_steps}"
    )

    optimizer = create_optimizer(
        model=model, use_pure_bf16=use_pure_bf16, use_main_grad=use_main_grad
    )
    if use_pure_bf16:
        level = 'O2'
        custom_white_list = None
        model = paddle.amp.decorate(
            models=model,
            dtype="bfloat16",
            level=level,
        )
    else:
        level = 'O1'
        custom_white_list = [
            "matmul_v2",
            "elementwise_add",
            "relu",
        ]
    model = paddle.DataParallel(model)

    if not use_pure_bf16:
        for param in model.parameters():
            t = paddle.cast(
                paddle.cast(param, dtype='bfloat16'), dtype='float32'
            )
            param.set_value(t)

    local_rank = paddle.distributed.get_rank()

    losses = []
    epoch = 2
    for eop in range(epoch):
        model.train()

        for batch_id, data in enumerate(train_loader()):
            data.stop_gradient = True

            enable_stats = False  # eop == 0
            if enable_stats:
                logging.info("<<<<<<<<<<<< forward-backward >>>>>>>>>>>")
                paddle.amp.debugging.enable_operator_stats_collection()
            with model.no_sync():
                with paddle.amp.auto_cast(
                    True,
                    level=level,
                    dtype="bfloat16",
                    custom_white_list=custom_white_list,
                ):
                    out = model(data)

                # compute loss in float32
                loss = paddle.mean(out.astype("float32"))

                # normal implementation for gradient accumulation.
                if acc_steps != 1:
                    loss = loss / acc_steps

                losses.append(loss.item())
                loss.backward()
                logging.info(
                    f"-- [rank={local_rank}] epoch {eop}, batch {batch_id}, loss: {loss.astype(paddle.float32).numpy()}"
                )
            if enable_stats:
                paddle.amp.debugging.disable_operator_stats_collection()

            if (batch_id + 1) % acc_steps == 0:
                if enable_stats:
                    logging.info(
                        "<<<<<<<<<<<< fused_allreduce_gradients >>>>>>>>>>>"
                    )
                    paddle.amp.debugging.enable_operator_stats_collection()
                fused_allreduce_gradients(list(model.parameters()), None)
                if enable_stats:
                    paddle.amp.debugging.disable_operator_stats_collection()

                if enable_stats:
                    logging.info("<<<<<<<<<<<< optimizer >>>>>>>>>>>")
                    paddle.amp.debugging.enable_operator_stats_collection()
                optimizer.step()
                optimizer.clear_grad()
                if enable_stats:
                    paddle.amp.debugging.disable_operator_stats_collection()

    model_param_dict = save_model_parameters(model)
    optimizer_state_dict = optimizer.state_dict()
    return losses, model_param_dict, optimizer_state_dict


def test_dp_bf16():
    if not paddle.amp.is_bfloat16_supported():
        logging.info("BFloat16 is not supported!")
        return

    paddle.distributed.init_parallel_env()
    local_rank = paddle.distributed.get_rank()
    paddle.seed(2023 + local_rank)
    np.random.seed(2023 + local_rank)

    # For DataParallel, DataLoader should feed different data for different GPUs.
    train_loader = paddle.io.DataLoader(
        RandomDataset(),
        batch_size=100,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    single_mlp = MLP()
    state_dict = single_mlp.state_dict()

    def _compare_bf16_o1_vs_o2(acc_steps=1):
        # dp bf16 O1 vs dp bf16 O2 main_grad
        mlp1 = MLP()
        mlp2 = MLP()
        mlp1.set_state_dict(state_dict)
        mlp2.set_state_dict(state_dict)
        losses_o1, model_param_dict_o1, optimizer_state_dict_o1 = train_mlp(
            mlp1, train_loader, use_pure_bf16=False, acc_steps=acc_steps
        )
        losses_o2, model_param_dict_o2, optimizer_state_dict_o2 = train_mlp(
            mlp2,
            train_loader,
            use_pure_bf16=True,
            use_main_grad=True,
            acc_steps=acc_steps,
        )
        np.testing.assert_array_equal(losses_o2, losses_o1)
        compare_state_dict(
            model_param_dict_o1, model_param_dict_o2, optimizer_state_dict_o2
        )

    # no gradient accumulation
    _compare_bf16_o1_vs_o2(acc_steps=1)
    # gradient accumulation
    _compare_bf16_o1_vs_o2(acc_steps=2)


if __name__ == '__main__':
    test_dp_bf16()
