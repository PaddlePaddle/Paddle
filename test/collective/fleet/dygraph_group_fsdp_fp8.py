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

import os
import shutil
import tempfile

import numpy as np
from dygraph_group_fsdp import ACCUM_STEPS, TransformerLayer

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fsdp._fp8_utils import sync_ckpt_weight_to_main
from paddle.distributed.fsdp.fully_shard import fully_shard
from paddle.optimizer.muon import Muon, MuonParamInfo

EXPERTS, HIDDEN, INTER, TOKENS, LAYERS, STEPS = 4, 128, 128, 256, 3, 4
BLOCK = 128
FP8_ATTRS = (
    ("fp8_weight", "fp8_scale", False),
    ("fp8_weight_t", "fp8_scale_t", True),
)


def rand_data(seed, steps):
    paddle.seed(seed)
    return [paddle.randn([TOKENS, HIDDEN]) for _ in range(steps * ACCUM_STEPS)]


def block_quant(weights, transpose):
    w = weights.astype("float32")
    if transpose:
        w = w.transpose([0, 2, 1])
    num, rows, cols = w.shape
    blocks = w.reshape([num, rows // BLOCK, BLOCK, cols // BLOCK, BLOCK])
    scale = blocks.abs().max(axis=[2, 4], keepdim=True).clip(min=1e-12) / 448.0
    fp8 = (blocks / scale).astype(paddle.float8_e4m3fn)
    return (
        fp8.reshape([num * rows, cols]),
        scale.reshape([num * rows // BLOCK, cols // BLOCK]),
    )


def BlockQuantizer(keep_transpose):
    def quantize(weights, transpose):
        return block_quant(weights, transpose)

    quantize.block = BLOCK
    quantize.attrs = FP8_ATTRS if keep_transpose else FP8_ATTRS[:1]
    return quantize


def dequant(fp8, scale):
    s = paddle.repeat_interleave(scale, BLOCK, axis=0)
    out = fp8.astype("float32") * paddle.repeat_interleave(s, BLOCK, axis=1)
    return out.reshape([EXPERTS, -1, out.shape[1]])


class FP8ExpertGEMM(PyLayer):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x)
        ctx.weight = weight
        if getattr(weight, "fp8_weight_t", None) is not None:
            w_nk = dequant(weight.fp8_weight_t, weight.fp8_scale_t)
        else:
            w_nk = dequant(weight.fp8_weight, weight.fp8_scale).transpose(
                [0, 2, 1]
            )
        return paddle.matmul(x, w_nk, transpose_y=True)

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensor()
        w = dequant(ctx.weight.fp8_weight, ctx.weight.fp8_scale)
        dx = paddle.matmul(dy, w, transpose_y=True)
        dw = paddle.matmul(x, dy, transpose_x=True)
        return dx, dw.astype("bfloat16")


class GroupedExpertLayer(TransformerLayer):
    def __init__(self, fp8):
        nn.Layer.__init__(self)
        self.gemm = FP8ExpertGEMM.apply if fp8 else paddle.matmul
        self.attn = nn.Linear(HIDDEN, HIDDEN, bias_attr=False)
        self.weight1 = self.create_parameter(
            [EXPERTS, HIDDEN, 2 * INTER], dtype="bfloat16"
        )
        self.weight2 = self.create_parameter(
            [EXPERTS, INTER, HIDDEN], dtype="bfloat16"
        )

    def forward(self, x):
        x = x + self.attn(x)
        h = self.gemm(x.reshape([EXPERTS, -1, HIDDEN]), self.weight1)
        h = F.silu(h[..., :INTER]) * h[..., INTER:]
        return x + 0.1 * self.gemm(h, self.weight2).reshape([-1, HIDDEN])


class GroupedExpertModel(nn.Layer):
    def __init__(self, fp8):
        super().__init__()
        self.embed = nn.Linear(HIDDEN, HIDDEN, bias_attr=False)
        self.layers = nn.LayerList(
            [GroupedExpertLayer(fp8) for _ in range(LAYERS)]
        )
        self.head = nn.Linear(HIDDEN, 1, bias_attr=False)
        # Alone in its (bf16) buffer, so rank 1's shard is only padding.
        self.scale = self.create_parameter(
            [1],
            dtype="bfloat16",
            default_initializer=nn.initializer.Constant(1.0),
        )

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.head(x)
        return x * self.scale.astype(x.dtype)


def experts(model):
    return [w for layer in model.layers for w in (layer.weight1, layer.weight2)]


def build_grouped(fp8, quantizer=None, muon=False, multi_precision=True):
    group = fleet.get_hybrid_communicate_group().get_sharding_parallel_group()
    paddle.seed(2026)
    model = GroupedExpertModel(fp8)
    for p in experts(model):
        p.is_moe_param = True
        p.expert = True
        p.color = {"color": "moe_expert", "group": group}
        if quantizer is not None:
            p._fsdp_fp8_quantizer = quantizer
    params = model.parameters()
    if muon:
        names = {p.name for p in experts(model)}
        for p in params:
            p.use_muon = p.name in names
        opt = Muon(
            learning_rate=1e-3,
            parameters=params,
            weight_decay=0.0,
            muon_param_info_map={
                p.name: MuonParamInfo(use_muon=p.use_muon) for p in params
            },
            ns_per_matrix=True,
            multi_precision=multi_precision,
        )
    else:
        opt = paddle.optimizer.AdamW(
            learning_rate=1e-3,
            parameters=params,
            weight_decay=0.0,
            multi_precision=multi_precision,
        )
    model = fully_shard(
        model,
        optimizer=opt,
        grad_accum_steps=ACCUM_STEPS,
    )
    return model, opt, model._fsdp_context


def train_steps(model, opt, data, steps):
    losses = []
    for step in steps:
        for x in data[step * ACCUM_STEPS :][:ACCUM_STEPS]:
            loss = (model(x) ** 2).mean()
            losses.append(loss._md5sum())
            loss.backward()
        opt.step()
        opt.clear_grad()
    return losses


@paddle.no_grad()
def expected_caches(ctx, model, quantizer):
    bm, fulls, md5s = ctx.buffer_manager, {}, []
    for layer in model.layers:
        for w in (layer.weight1, layer.weight2):
            buf = bm.buffer_groups[bm.param_to_buffer_id[w.name]].params_buffer
            if buf.unique_key not in fulls:
                parts = []
                paddle.distributed.all_gather(
                    parts, buf.data_buffer, group=buf.fsdp_group
                )
                fulls[buf.unique_key] = paddle.concat(parts)
            begin = buf.param_offsets[w.name]
            full = fulls[buf.unique_key][begin : begin + int(np.prod(w.shape))]
            full = full.astype(w.dtype).reshape(w.shape)
            for _, _, transpose in quantizer.attrs:
                md5s += [t._md5sum() for t in block_quant(full, transpose)]
    return md5s


def train_fp8(keep_transpose, fp8_gather, data, muon=False, mp=True):
    quantizer = BlockQuantizer(keep_transpose)
    model, opt, ctx = build_grouped(
        True,
        quantizer=quantizer if fp8_gather else None,
        muon=muon,
        multi_precision=mp,
    )
    groups = ctx.buffer_manager.buffer_groups
    fp8_groups = [g for g in groups if g.params_buffer.fp8 is not None]
    assert len(fp8_groups) == LAYERS * fp8_gather
    if muon:
        assert all(g.use_muon and g.muon_shard_numel for g in fp8_groups)
    ctx.all_gather_params()

    attrs = [a for d, s, _ in quantizer.attrs for a in (d, s)]
    n_caches = len(experts(model)) * len(attrs)

    def record_caches(layer, *_):
        # Within the unit: caches are released once it is done.
        if len(caches[-1]) < n_caches:
            caches[-1] += [
                getattr(w, a)._md5sum()
                for w in (layer.weight1, layer.weight2)
                for a in attrs
            ]

    for layer in model.layers:
        layer.register_forward_post_hook(record_caches)
    losses, caches = [], []
    for step in range(STEPS):
        if not fp8_gather:  # reference: quantize the full bf16 weight
            with paddle.no_grad(), ctx.unshard_params(experts(model)):
                for w in experts(model):
                    for d, s, transpose in quantizer.attrs:
                        fp8, scale = block_quant(w, transpose)
                        setattr(w, d, fp8)
                        setattr(w, s, scale)
        else:
            expected = expected_caches(ctx, model, quantizer)
        caches.append([])
        losses += train_steps(model, opt, data, [step])
        if fp8_gather:
            assert caches[-1] == expected, f"step {step}: fp8 caches differ"
            assert not hasattr(model.layers[0].weight1, "fp8_weight")
            for g in fp8_groups:  # fp32 main iff an fp32 master; no master
                main = g.params_buffer.data_buffer
                main_dtype = paddle.float32 if mp else g.dtype
                assert main.dtype == main_dtype, main.dtype
                names = [main.name] + [p.name for p in g.params]
                assert not set(names) & set(opt._master_weights)
                assert all(p.dtype == g.dtype for p in g.params)
    if fp8_gather:
        try:
            with ctx.unshard_params(list(model.parameters())):
                pass
        except RuntimeError:
            pass
        else:
            raise AssertionError("unshard_params must refuse fp8 groups")
    return losses, caches


def run_fp8_gather():
    """Shard quantization + fp8 gather == quantizing the full weight."""
    data = rand_data(7, STEPS)
    for keep_transpose, muon, mp in (
        (True, False, True),
        (False, False, True),
        (True, True, True),
        (True, False, False),
    ):
        ref = train_fp8(keep_transpose, False, data, muon, mp)
        new = train_fp8(keep_transpose, True, data, muon, mp)
        tag = f"{keep_transpose=} {muon=} {mp=}"
        # Only step 0 across runs: after an optimizer step, bf16 param +
        # master vs fp32 main may round differently on some archs (Volta).
        assert ref[1][0] == new[1][0], f"{tag}: fp8 caches differ"
        assert ref[0][:ACCUM_STEPS] == new[0][:ACCUM_STEPS], tag


def fp8_slices(ctx):
    for group in ctx.buffer_manager.buffer_groups:
        if group.params_buffer.fp8 is not None:
            main = group.params_buffer.data_buffer
            for key, info in ctx._shard_descs[main.name].items():
                yield key, info, main


def check_fp8_checkpoint(ckpt_path, muon, mp=True):
    data = rand_data(7, 3)

    def build():
        return build_grouped(
            True,
            quantizer=BlockQuantizer(True),
            muon=muon,
            multi_precision=mp,
        )

    model, opt, ctx = build()
    train_steps(model, opt, data, [0, 1])
    # A padding-only shard gets no optimizer state.
    padding = [
        g.params_buffer.data_buffer
        for g in ctx.buffer_manager.buffer_groups
        if not any(g.params_buffer.param_shard(p) for p in g.params)
    ]
    assert padding or paddle.distributed.get_rank() == 0
    for buf in padding:
        assert buf.stop_gradient
        assert all(buf.name not in a for a in opt._accumulators.values())
    state = dict(model.sharded_state_dict())
    state.update(opt.sharded_state_dict())
    keys = [key for key, _, _ in fp8_slices(ctx)]
    assert keys, "no fp8 shard on this rank"
    for key in keys:
        weight = state[key].local_tensor
        assert weight.dtype == paddle.bfloat16, (key, weight.dtype)
        if not mp:  # bf16 main weight, no master to save
            assert f"{key}.w_0" not in state, key
            continue
        master = state[f"{key}.w_0"].local_tensor
        assert master.dtype == paddle.float32, (key, master.dtype)
        np.testing.assert_array_equal(
            weight.numpy(), master.astype("bfloat16").numpy(), err_msg=key
        )
    saved = {
        k: v.local_tensor.astype("float32").numpy() for k, v in state.items()
    }
    paddle.distributed.save_state_dict(state, ckpt_path)
    paddle.distributed.barrier()
    expected = train_steps(model, opt, data, [2])

    model, opt, ctx = build()
    opt.init_state_for_load()
    state = dict(model.sharded_state_dict())
    state.update(opt.sharded_state_dict())
    assert sorted(state) == sorted(saved), sorted(set(state) ^ set(saved))
    # Another call before loading (as EMA does) must reuse the fp8 copies.
    model.sharded_state_dict()
    paddle.distributed.load_state_dict(state, ckpt_path)
    for key, sharded in state.items():
        np.testing.assert_array_equal(
            sharded.local_tensor.astype("float32").numpy(), saved[key], key
        )
    assert train_steps(model, opt, data, [2]) == expected, f"{muon=}"

    model, opt, ctx = build()
    state = dict(model.sharded_state_dict())
    paddle.distributed.load_state_dict(state, ckpt_path)
    for group in ctx.buffer_manager.buffer_groups:
        if group.params_buffer.fp8 is not None:
            sync_ckpt_weight_to_main(group.params_buffer)
    for key, info, main in fp8_slices(ctx):
        restored = ctx._create_sharded_weight(key, main, info).local_tensor
        np.testing.assert_array_equal(
            restored.astype("bfloat16").astype("float32").numpy(),
            saved[key],
            err_msg=key,
        )


class OddExpertLayer(TransformerLayer):
    # 3 experts: weight2 is 1.5 alignment units, so the buffer can't shard.
    def __init__(self):
        nn.Layer.__init__(self)
        self.weight1 = self.create_parameter([3, HIDDEN, 2 * INTER])
        self.weight2 = self.create_parameter([3, INTER, HIDDEN])


def check_unaligned_fp8_raises():
    layer = OddExpertLayer()
    quantizer = BlockQuantizer(True)
    for p in (layer.weight1, layer.weight2):
        p.expert = True
        p._fsdp_fp8_quantizer = quantizer
    try:
        fully_shard(nn.LayerList([layer]))
    except ValueError as e:
        assert "is not sharded" in str(e), e
    else:
        raise AssertionError("unaligned fp8 group must be refused")


def run_fp8_checkpoint():
    rank = paddle.distributed.get_rank()
    ckpt_root = [tempfile.mkdtemp() if rank == 0 else None]
    paddle.distributed.broadcast_object_list(ckpt_root, src=0)
    ckpt_root = ckpt_root[0]
    try:
        for muon, mp in ((False, True), (True, True), (False, False)):
            path = os.path.join(ckpt_root, f"{muon=}_{mp=}")
            check_fp8_checkpoint(path, muon, mp)
    finally:
        paddle.distributed.barrier()
        if rank == 0:
            shutil.rmtree(ckpt_root, ignore_errors=True)


if __name__ == '__main__':
    paddle.distributed.init_parallel_env()
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": paddle.distributed.get_world_size(),
    }
    fleet.init(is_collective=True, strategy=strategy)
    check_unaligned_fp8_raises()
    run_fp8_gather()
    run_fp8_checkpoint()
