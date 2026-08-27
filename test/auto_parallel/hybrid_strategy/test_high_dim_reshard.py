# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import os
import tempfile
import unittest

import collective.test_communication_api_base as test_base

os.environ["FLAGS_enable_pir_api"] = "0"


class TestHighDimReshard(test_base.CommunicationTestDistBase):
    """End-to-end two-device FlexCheckpoint test for high-dimensional slices."""

    def test_high_dim_reshard_two_devices(self):
        with tempfile.TemporaryDirectory() as ckpt_dir:
            envs = {
                "device_num": "2",
                "high_dim_ckpt_path": ckpt_dir,
                "high_dim_mode": "save",
            }
            super().setUp(num_of_devices=2, timeout=180, nnode=1)
            self.run_test_case("high_dim_reshard_worker.py", envs)

            for comm_method in ("broadcast", "send_recv", "grouped_send_recv"):
                envs = {
                    "device_num": "2",
                    "high_dim_ckpt_path": ckpt_dir,
                    "high_dim_mode": "load",
                    "high_dim_comm_method": comm_method,
                }
                super().setUp(num_of_devices=2, timeout=180, nnode=1)
                self.run_test_case("high_dim_reshard_worker.py", envs)


if __name__ == "__main__":
    unittest.main()
