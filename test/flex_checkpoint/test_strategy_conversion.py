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

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import unittest

import paddle


def p_str_to_dict(p_str):
    """Parses a strategy string like 'd2·t2' into a config dictionary."""
    config = {"tp": 1, "dp": 1, "pp": 1, "ep": 1}
    parts = p_str.split('·')
    for part in parts:
        if part.startswith('d'):
            config['dp'] = int(part[1:])
        elif part.startswith('t'):
            config['tp'] = int(part[1:])
        elif part.startswith('p'):
            config['pp'] = int(part[1:])
        elif part.startswith('e'):
            config['ep'] = int(part[1:])

    if config['ep'] > 1 and config['dp'] < config['ep']:
        config['dp'] = config['ep']

    config["num_cards"] = config["tp"] * config["dp"] * config["pp"]
    if p_str in ["d1", "t1", "p1", "e1"]:
        config["num_cards"] = 1

    return config


TEST_CASES = [
    {
        "id": "B1_d2_to_d4",
        "src": p_str_to_dict("d2"),
        "tgt": p_str_to_dict("d4"),
        "gpu_num": 4,
    },
    {
        "id": "B2_t2_to_t4",
        "src": p_str_to_dict("t2"),
        "tgt": p_str_to_dict("t4"),
        "gpu_num": 4,
    },
    {
        "id": "B3_p2_to_p4",
        "src": p_str_to_dict("p2"),
        "tgt": p_str_to_dict("p4"),
        "gpu_num": 4,
    },
    {
        "id": "B4_e2_to_e4",
        "src": p_str_to_dict("e2"),
        "tgt": p_str_to_dict("e4"),
        "model_type": "moe",
        "gpu_num": 4,
    },
    # Case 5 (pp2 -> tp4)
    {
        "id": "X5_pp2_to_tp4",
        "src": p_str_to_dict("p2"),
        "tgt": p_str_to_dict("t4"),
        "gpu_num": 4,
    },
    # Case 6 (tp2 -> pp2)
    {
        "id": "X6_tp2_to_pp2",
        "src": p_str_to_dict("t2"),
        "tgt": p_str_to_dict("p2"),
        "gpu_num": 2,
    },
    # Case 7 (dp4 -> tp2·dp2)
    {
        "id": "X7_dp4_to_tp2dp2",
        "src": p_str_to_dict("d4"),
        "tgt": p_str_to_dict("t2·d2"),
        "gpu_num": 4,
    },
    # Case 8 (dp2 -> pp2)
    {
        "id": "X8_dp2_to_pp2",
        "src": p_str_to_dict("d2"),
        "tgt": p_str_to_dict("p2"),
        "gpu_num": 2,
    },
    # Case 9 (dp2 -> ep2)
    {
        "id": "X9_dp2_to_ep2",
        "src": p_str_to_dict("d2"),
        "tgt": p_str_to_dict("e2"),
        "model_type": "moe",
        "gpu_num": 2,
    },
    # Case 10 (ep2 -> tp2)
    {
        "id": "X10_ep2_to_tp2",
        "src": p_str_to_dict("e2"),
        "tgt": p_str_to_dict("t2"),
        "model_type": "moe",
        "gpu_num": 2,
    },
    # Case 11 (tp2 -> ep2)
    {
        "id": "X11_tp2_to_ep2",
        "src": p_str_to_dict("t2"),
        "tgt": p_str_to_dict("e2"),
        "model_type": "moe",
        "gpu_num": 2,
    },
    {
        "id": "M12_dp2tp2_to_tp4",
        "src": p_str_to_dict("d2·t2"),
        "tgt": p_str_to_dict("t4"),
        "gpu_num": 4,
    },
    {
        "id": "M13_dp2tp2_to_pp4",
        "src": p_str_to_dict("d2·t2"),
        "tgt": p_str_to_dict("p4"),
        "gpu_num": 4,
    },
    {
        "id": "M14_dp2pp2_to_tp4",
        "src": p_str_to_dict("d2·p2"),
        "tgt": p_str_to_dict("t4"),
        "gpu_num": 4,
    },
    {
        "id": "M15_tp2pp2_to_dp4",
        "src": p_str_to_dict("t2·p2"),
        "tgt": p_str_to_dict("d4"),
        "gpu_num": 4,
    },
    {
        "id": "M16_tp2pp2_to_dp2tp2",
        "src": p_str_to_dict("t2·p2"),
        "tgt": p_str_to_dict("d2·t2"),
        "gpu_num": 4,
    },
    {
        "id": "M17_dp2ep2_to_dp4",
        "src": p_str_to_dict("d2·e2"),
        "tgt": p_str_to_dict("d4"),
        "model_type": "moe",
        "gpu_num": 4,
    },
    {
        "id": "M18_tp2ep2_to_tp4",
        "src": p_str_to_dict("t2·e2"),
        "tgt": p_str_to_dict("t4"),
        "model_type": "moe",
        "gpu_num": 4,
    },
    # Case 19 (dp2·tp2 -> pp2)
    {
        "id": "M19_dp2tp2_to_pp2",
        "src": p_str_to_dict("d2·t2"),
        "tgt": p_str_to_dict("p2"),
        "gpu_num": 4,
    },
    # E1 (e2->e4) is covered by B4
    {
        "id": "E2_dp2ep2_to_tp2ep2",
        "src": p_str_to_dict("d2·e2"),
        "tgt": p_str_to_dict("t2·e2"),
        "model_type": "moe",
        "gpu_num": 4,
    },
]


class TestStrategyConversion(unittest.TestCase):
    def _run_workflow(self, case, logic_script="strategy_conversion_engine.py"):
        if case["gpu_num"] > paddle.device.cuda.device_count():
            self.skipTest("number of GPUs is not enough")

        case_id = case['id']
        src_config = case['src']
        tgt_config = case['tgt']

        src_gpus_count = src_config.pop("num_cards")
        tgt_gpus_count = tgt_config.pop("num_cards")
        src_gpus = ",".join(map(str, range(src_gpus_count)))
        tgt_gpus = ",".join(map(str, range(tgt_gpus_count)))

        with tempfile.TemporaryDirectory() as tmpdir:
            src_ckpt_path = os.path.join(tmpdir, "src_ckpt")
            tgt_ckpt_path = os.path.join(tmpdir, "tgt_ckpt")

            def config_to_args(config, prefix):
                return [
                    f"--{prefix}_{k}={v}"
                    for k, v in config.items()
                    if not k.startswith('s_')
                ]

            common_args = config_to_args(src_config, "src") + config_to_args(
                tgt_config, "tgt"
            )
            if "model_type" in case:
                common_args.append(f"--model_type={case['model_type']}")
            path_args = [
                f"--src_ckpt_path={src_ckpt_path}",
                f"--tgt_ckpt_path={tgt_ckpt_path}",
            ]
            base_cmd = [
                sys.executable,
                "-m",
                "paddle.distributed.launch",
                "--log_dir",
                os.path.join(tmpdir, "logs"),
            ]

            steps = ["save_source", "convert", "verify"]
            gpus_per_step = [src_gpus, tgt_gpus, src_gpus]

            for i, step_name in enumerate(steps):
                cmd = [
                    *base_cmd,
                    f"--gpus={gpus_per_step[i]}",
                    logic_script,
                    f"--step={step_name}",
                    *common_args,
                    *path_args,
                ]
                process = subprocess.run(
                    cmd, capture_output=True, text=True, check=False
                )

                self.assertEqual(
                    process.returncode,
                    0,
                    f"Step '{step_name}' FAILED for case '{case_id}'!\n"
                    f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}",
                )


def _create_test_method(case):
    def test_method(self):
        self._run_workflow(case)

    return test_method


for case_info in TEST_CASES:
    test_name = f"test_{case_info['id']}"
    test_func = _create_test_method(case_info)
    setattr(TestStrategyConversion, test_name, test_func)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--list_tests',
        action='store_true',
        help='List all test case names that unittest can discover and exit.',
    )
    args, unknown = parser.parse_known_args()

    if args.list_tests:
        for case in TEST_CASES:
            module_name = os.path.splitext(os.path.basename(__file__))[0]
            logging.basicConfig(
                stream=sys.stdout, level=logging.INFO, format="%(message)s"
            )
            logging.info(
                f"{module_name}.TestStrategyConversion.test_{case['id']}"
            )
        sys.exit(0)

    unittest.main(argv=[sys.argv[0]], *unknown)
