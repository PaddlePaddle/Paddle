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

import json
import os
import subprocess
import sys
import textwrap
import unittest

import paddle

POOL_STABLE = 0
POOL_LONGLIVED = 1

# The BestFit allocator aligns each allocation to GpuMinChunkSize (256 bytes).
# block.size_ reflects the aligned size, so expected byte totals must account
# for per-tensor alignment, not just raw element counts.
VMM_ALIGNMENT = 256


def aligned_size(size):
    """Round *size* up to the next multiple of VMM_ALIGNMENT."""
    rem = size % VMM_ALIGNMENT
    return size if rem == 0 else size + VMM_ALIGNMENT - rem


@unittest.skipIf(
    (not paddle.is_compiled_with_cuda()) or paddle.is_compiled_with_rocm(),
    'should compile with cuda.',
)
class TestVMMAllocatorV2PoolHint(unittest.TestCase):
    def run_isolated_case(self, body: str):
        script = textwrap.dedent(
            f"""
            import json
            import paddle
            from paddle.base import core, framework

            POOL_STABLE = 0
            POOL_LONGLIVED = 1
            VMM_ALIGNMENT = 256

            def aligned_size(size):
                rem = size % VMM_ALIGNMENT
                return size if rem == 0 else size + VMM_ALIGNMENT - rem

            def collect_pool_stats():
                stats = {{}}
                for (
                    pool_type,
                    active_count,
                    active_bytes,
                    _free_count,
                    _free_bytes,
                    _gap_count,
                    _gap_bytes,
                ) in paddle.device.cuda.vmm_v2_pool_stats():
                    prev_count, prev_bytes = stats.get(pool_type, (0, 0))
                    stats[pool_type] = (
                        prev_count + active_count,
                        prev_bytes + active_bytes,
                    )
                return stats

            assert hasattr(core, "_get_vmm_pool_hint")
            assert hasattr(core, "_set_vmm_pool_hint")
            with framework.vmm_pool_hint_guard("stable"):
                assert core._get_vmm_pool_hint() == 1
            assert core._get_vmm_pool_hint() == 0

            {body}
            """
        )
        env = os.environ.copy()
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            self.fail(
                "Isolated VMM pool hint case failed.\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        return json.loads(result.stdout.strip())

    def test_training_routes_params_and_optimizer_state(self):
        result = self.run_isolated_case(
            """
            paddle.set_flags(
                {'FLAGS_use_vmm_auto_growth_best_fit_allocator_v2': True}
            )
            paddle.set_device('gpu')

            model = paddle.nn.Sequential(
                paddle.nn.Linear(16, 32),
                paddle.nn.ReLU(),
                paddle.nn.Linear(32, 4),
            )
            paddle.device.synchronize()
            after_param = collect_pool_stats()

            optimizer = paddle.optimizer.Adam(
                learning_rate=0.01, parameters=model.parameters()
            )
            x = paddle.randn([8, 16], dtype='float32')
            loss = model(x).mean()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            paddle.device.synchronize()
            after_step = collect_pool_stats()

            print(json.dumps({
                "after_param": after_param,
                "after_step": after_step,
            }))
            """
        )

        # Linear(16,32): weight 16*32*4=2048B, bias 32*4=128B
        # Linear(32,4):  weight 32*4*4=512B,   bias 4*4=16B
        # Each allocation is individually aligned to VMM_ALIGNMENT (256B).
        EXPECTED_PARAM_BLOCKS = 4
        EXPECTED_PARAM_BYTES = sum(
            aligned_size(s) for s in [16 * 32 * 4, 32 * 4, 32 * 4 * 4, 4 * 4]
        )
        EXPECTED_OPT_BLOCKS = 8
        EXPECTED_OPT_BYTES = EXPECTED_PARAM_BYTES * 2

        after_param = {
            int(k): tuple(v) for k, v in result["after_param"].items()
        }
        after_step = {int(k): tuple(v) for k, v in result["after_step"].items()}

        param_count, param_bytes = after_param.get(POOL_STABLE, (0, 0))
        self.assertEqual(
            param_count,
            EXPECTED_PARAM_BLOCKS,
            msg=(
                f"Stable pool: expected {EXPECTED_PARAM_BLOCKS} param blocks, "
                f"got {param_count}."
            ),
        )
        self.assertEqual(
            param_bytes,
            EXPECTED_PARAM_BYTES,
            msg=(
                f"Stable pool: expected {EXPECTED_PARAM_BYTES} param bytes, "
                f"got {param_bytes}."
            ),
        )
        stable_count, stable_bytes = after_step.get(POOL_STABLE, (0, 0))
        self.assertEqual(
            stable_count,
            param_count,
            msg=(
                f"Stable pool after step: expected to match after_param "
                f"count={param_count}, got {stable_count}."
            ),
        )
        self.assertEqual(
            stable_bytes,
            param_bytes,
            msg=(
                f"Stable pool after step: expected to match after_param "
                f"bytes={param_bytes}, got {stable_bytes}."
            ),
        )
        ll_count, ll_bytes = after_step.get(POOL_LONGLIVED, (0, 0))
        self.assertEqual(
            ll_count,
            EXPECTED_OPT_BLOCKS,
            msg=(
                f"LongLived pool: expected {EXPECTED_OPT_BLOCKS} optimizer "
                f"state blocks, got {ll_count}."
            ),
        )
        self.assertEqual(
            ll_bytes,
            EXPECTED_OPT_BYTES,
            msg=(
                f"LongLived pool: expected {EXPECTED_OPT_BYTES} optimizer "
                f"state bytes, got {ll_bytes}."
            ),
        )

    def test_amp_o2_routes_master_weight_and_moments(self):
        """AMP O2 + multi_precision: master weights → Stable, moments → LongLived.

        This is also a white-box regression test. The expected block counts and
        aligned byte totals are derived from the current AMP O2 implementation:
        amp.decorate() materializes fp16 parameters under the Stable hint, and
        Adam multi_precision creates one fp32 master weight plus two fp32
        moment tensors per parameter.
        """
        result = self.run_isolated_case(
            """
            paddle.set_flags(
                {'FLAGS_use_vmm_auto_growth_best_fit_allocator_v2': True}
            )
            paddle.set_device('gpu')

            model = paddle.nn.Sequential(
                paddle.nn.Linear(16, 32),
                paddle.nn.ReLU(),
                paddle.nn.Linear(32, 4),
            )
            model = paddle.amp.decorate(models=model, level='O2')
            paddle.device.synchronize()
            after_decorate = collect_pool_stats()
            optimizer = paddle.optimizer.Adam(
                learning_rate=0.01,
                parameters=model.parameters(),
                multi_precision=True,
            )

            scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
            with paddle.amp.auto_cast(level='O2'):
                x = paddle.randn([8, 16], dtype='float32')
                loss = model(x).mean()
            scaled = scaler.scale(loss)
            scaled.backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.clear_grad()
            paddle.device.synchronize()

            print(json.dumps({
                "after_decorate": after_decorate,
                "after_step": collect_pool_stats(),
            }))
            """
        )
        after_decorate = {
            int(k): tuple(v) for k, v in result["after_decorate"].items()
        }
        after_step = {int(k): tuple(v) for k, v in result["after_step"].items()}

        # Float32 param sizes (raw), each individually aligned to 256B:
        #   w1(16*32*4=2048→2048) + b1(32*4=128→256)
        #   w2(32*4*4=512→512)    + b2(4*4=16→256)  → total 3072B
        PARAM_BYTES_F32 = sum(
            aligned_size(s) for s in [16 * 32 * 4, 32 * 4, 32 * 4 * 4, 4 * 4]
        )  # 3072
        # Float16 param sizes (raw), each individually aligned to 256B:
        #   w1(16*32*2=1024→1024) + b1(32*2=64→256)
        #   w2(32*4*2=256→256)    + b2(4*2=8→256)   → total 1792B
        PARAM_BYTES_F16 = sum(
            aligned_size(s) for s in [16 * 32 * 2, 32 * 2, 32 * 4 * 2, 4 * 2]
        )  # 1792

        decorate_stable_count, decorate_stable_bytes = after_decorate.get(
            POOL_STABLE, (0, 0)
        )
        self.assertEqual(
            decorate_stable_count,
            4,
            msg=(
                "Stable pool after decorate: expected 4 fp16 parameter blocks, "
                f"got {decorate_stable_count}."
            ),
        )
        self.assertEqual(
            decorate_stable_bytes,
            PARAM_BYTES_F16,
            msg=(
                "Stable pool after decorate: expected only fp16 parameter "
                f"bytes={PARAM_BYTES_F16}, got {decorate_stable_bytes}."
            ),
        )

        # Stable pool after O2 step under the current implementation:
        #   1. Original float32 params are created in Stable, then replaced by
        #      amp.decorate(), so they should not remain active here.
        #   2. New float16 params (cast by amp.decorate, under stable hint):
        #      4 blocks totaling PARAM_BYTES_F16.
        #   3. master_weight (fp32 copy created by Adam multi_precision, under
        #      stable hint): 4 blocks totaling PARAM_BYTES_F32.
        #   Total: 8 active Stable blocks, 1792 + 3072 = 4864B.
        EXPECTED_STABLE_BLOCKS = 8
        EXPECTED_STABLE_BYTES = PARAM_BYTES_F16 + PARAM_BYTES_F32  # 4864

        stable_count, stable_bytes = after_step.get(POOL_STABLE, (0, 0))
        self.assertEqual(
            stable_count,
            EXPECTED_STABLE_BLOCKS,
            msg=(
                f"Stable pool: expected {EXPECTED_STABLE_BLOCKS} blocks "
                f"(4 fp16 params + 4 master_weights), "
                f"got {stable_count}."
            ),
        )
        self.assertEqual(
            stable_bytes,
            EXPECTED_STABLE_BYTES,
            msg=(
                f"Stable pool: expected {EXPECTED_STABLE_BYTES} bytes "
                f"(fp16 {PARAM_BYTES_F16} + master_weight {PARAM_BYTES_F32}), "
                f"got {stable_bytes}."
            ),
        )

        # LongLived pool under the current implementation: only Adam moment1
        # and moment2 are routed here. That is 4 parameters × 2 tensors = 8
        # active blocks, with total aligned bytes equal to fp32 param bytes × 2.
        EXPECTED_LL_BLOCKS = 8
        EXPECTED_LL_BYTES = PARAM_BYTES_F32 * 2  # 6144

        ll_count, ll_bytes = after_step.get(POOL_LONGLIVED, (0, 0))
        self.assertEqual(
            ll_count,
            EXPECTED_LL_BLOCKS,
            msg=(
                f"LongLived pool: expected {EXPECTED_LL_BLOCKS} blocks "
                f"(4 params × [moment1+moment2]), "
                f"got {ll_count}."
            ),
        )
        self.assertEqual(
            ll_bytes,
            EXPECTED_LL_BYTES,
            msg=(
                f"LongLived pool: expected {EXPECTED_LL_BYTES} bytes, "
                f"got {ll_bytes}."
            ),
        )


if __name__ == '__main__':
    unittest.main()
