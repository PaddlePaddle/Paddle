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

import unittest
from functools import partial

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestXPUFusedMultiTransformerCacheKVLayoutTransPass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["fuse_multi_transformer_cachekv_layout_trans"], (
            1e-3,
            1e-3,
        )

    def sample_program_config(self, draw):

        # shape + slice input
        src_ids_shape = draw(st.sampled_from([[1, 64], [2, 64]]))
        # fused_multi_transformer
        layers_num = draw(st.sampled_from([1]))
        x_shape = draw(st.sampled_from([[1, 128, 1024]]))
        src_mask_shape = draw(st.sampled_from([[1, 16, 128, 128]]))
        ln_scale_shape = draw(st.sampled_from([[1024]]))
        ln_bias_shape = draw(st.sampled_from([[1024]]))
        ffn_ln_scale_shape = draw(st.sampled_from([[1024]]))
        ffn_ln_bias_shape = draw(st.sampled_from([[1024]]))
        qkv_w_shape = draw(st.sampled_from([[3, 16, 64, 1024]]))
        out_linear_w_shape = draw(st.sampled_from([[1024, 1024]]))
        ffn1_w_shape = draw(st.sampled_from([[1024, 4096]]))
        ffn2_w_shape = draw(st.sampled_from([[4096, 1024]]))
        qkv_bias_shape = draw(st.sampled_from([[3072]]))
        out_linear_bias_shape = draw(st.sampled_from([[1024]]))
        ffn1_bias_shape = draw(st.sampled_from([[4096]]))
        ffn2_bias_shape = draw(st.sampled_from([[1024]]))
        cache_kv_shape = draw(st.sampled_from([[2, -1, 16, 1024, 64]]))

        ops = []

        def gen_shape_slice_ops():
            shape_op_config = OpConfig(
                "shape",
                inputs={
                    "Input": ["src_ids"],
                },
                outputs={"Out": ["shape_out"]},
            )
            slice_op_config = OpConfig(
                "slice",
                inputs={
                    "Input": ["shape_out"],
                },
                outputs={"Out": ["slice_out"]},
                axes=0,
                decrease_axis=0,
                ends=1,
                infer_flags=1,
                starts=0,
            )
            return [shape_op_config, slice_op_config]

        def gen_fill_constant_ops():
            fill_constant_op_0 = OpConfig(
                type="fill_constant",
                inputs={},
                outputs={"Out": ["fill_constant_0_out"]},
                attrs={
                    "dtype": 2,
                    "str_value": cache_kv_shape[0],
                    "value": cache_kv_shape[0],
                    "shape": [1],
                },
            )
            fill_constant_op_1 = OpConfig(
                type="fill_constant",
                inputs={},
                outputs={"Out": ["fill_constant_1_out"]},
                attrs={
                    "dtype": 2,
                    "str_value": cache_kv_shape[2],
                    "value": cache_kv_shape[2],
                    "shape": [1],
                },
            )
            fill_constant_op_2 = OpConfig(
                type="fill_constant",
                inputs={},
                outputs={"Out": ["fill_constant_2_out"]},
                attrs={
                    "dtype": 2,
                    "str_value": cache_kv_shape[3],
                    "value": cache_kv_shape[3],
                    "shape": [1],
                },
            )
            fill_constant_op_3 = OpConfig(
                type="fill_constant",
                inputs={},
                outputs={"Out": ["fill_constant_3_out"]},
                attrs={
                    "dtype": 2,
                    "str_value": cache_kv_shape[4],
                    "value": cache_kv_shape[4],
                    "shape": [1],
                },
            )
            return [
                fill_constant_op_0,
                fill_constant_op_1,
                fill_constant_op_2,
                fill_constant_op_3,
            ]

        def gen_fill_constant_reduce():
            fill_constant_op_reduce = OpConfig(
                type="fill_constant",
                inputs={
                    "ShapeTensorList": [
                        "fill_constant_0_out",
                        "slice_out",
                        "fill_constant_1_out",
                        "fill_constant_2_out",
                        "fill_constant_3_out",
                    ]
                },
                outputs={"Out": ["fill_constant_reduce_out"]},
                attrs={
                    "dtype": 4,
                    "str_value": 0,
                    "value": 0,
                },
            )
            return [fill_constant_op_reduce]

        def gen_fused_multi_transformer_op():
            fused_multi_transformer_op = OpConfig(
                type="fused_multi_transformer",
                inputs={
                    "X": ["x"],
                    "CacheKV": ["fill_constant_reduce_out"],
                    "FFN1Bias": ["ffn1_bias"],
                    "FFN1Weight": ["ffn1_w"],
                    "FFN2Bias": ["ffn2_bias"],
                    "FFN2Weight": ["ffn2_w"],
                    "FFNLnBias": ["ffn_ln_bias"],
                    "FFNLnScale": ["ffn_ln_scale"],
                    "LnBias": ["ln_bias"],
                    "LnScale": ["ln_scale"],
                    "OutLinearBias": ["out_linear_bias"],
                    "OutLinearW": ["out_linear_w"],
                    "QKVBias": ["qkv_bias"],
                    "QKVW": ["qkv_w"],
                    "SrcMask": ["src_mask"],
                },
                outputs={
                    "Out": ["fused_multi_transformer_out"],
                    "CacheKVOut": ["fill_constant_reduce_out"],
                },
            )
            return [fused_multi_transformer_op]

        shape_slice_ops = gen_shape_slice_ops()
        fill_constant_ops = gen_fill_constant_ops()
        fill_constant_op_reduce = gen_fill_constant_reduce()
        fused_multi_transformer_op = gen_fused_multi_transformer_op()
        ops = []
        ops.extend(shape_slice_ops)
        ops.extend(fill_constant_ops)
        ops.extend(fill_constant_op_reduce)
        ops.extend(fused_multi_transformer_op)

        def generate_src_ids(*args, **kargs):
            return np.random.randint(0, 64, src_ids_shape).astype(np.int64)

        inputs = {
            "src_ids": TensorConfig(data_gen=partial(generate_src_ids)),
            "x": TensorConfig(shape=x_shape),
            "src_mask": TensorConfig(shape=src_mask_shape),
        }
        weights = {
            "ffn1_bias": TensorConfig(shape=ffn1_bias_shape),
            "ffn1_w": TensorConfig(shape=ffn1_w_shape),
            "ffn2_bias": TensorConfig(shape=ffn2_bias_shape),
            "ffn2_w": TensorConfig(shape=ffn2_w_shape),
            "ffn_ln_bias": TensorConfig(shape=ffn_ln_bias_shape),
            "ffn_ln_scale": TensorConfig(shape=ffn_ln_scale_shape),
            "ln_bias": TensorConfig(shape=ln_bias_shape),
            "ln_scale": TensorConfig(shape=ln_scale_shape),
            "out_linear_bias": TensorConfig(shape=out_linear_bias_shape),
            "out_linear_w": TensorConfig(shape=out_linear_w_shape),
            "qkv_bias": TensorConfig(shape=qkv_bias_shape),
            "qkv_w": TensorConfig(shape=qkv_w_shape),
        }
        program_config = ProgramConfig(
            ops=ops,
            weights=weights,
            inputs=inputs,
            outputs=["fused_multi_transformer_out", "fill_constant_reduce_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=3,
            min_success_num=3,
            passes=["fused_multi_transformer_cachekv_layout_trans_pass"],
        )


if __name__ == "__main__":
    unittest.main()
