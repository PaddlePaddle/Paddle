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

from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()

    # for distributed strategy
    parser.add_argument(
        "--dp_degree", type=int, required=True, help="dp degree"
    )
    parser.add_argument(
        "--mp_degree", type=int, required=True, help="mp degree"
    )
    parser.add_argument(
        "--pp_degree", type=int, required=True, help="pp degree"
    )
    parser.add_argument(
        "--vpp_degree", type=int, required=True, help="vpp degree"
    )
    parser.add_argument(
        "--sharding_degree", type=int, required=True, help="sharding degree"
    )
    parser.add_argument(
        "--sharding_stage", type=int, required=True, help="sharding stage"
    )
    parser.add_argument(
        "--micro_batch_size", type=int, required=True, help="micro batch size"
    )
    parser.add_argument(
        "--use_recompute", type=bool, required=True, help="use recompute"
    )
    parser.add_argument(
        "--recompute_granularity",
        type=str,
        required=True,
        choices=["None", "core_attn", "full_attn", "full"],
        help="recompute granularity",
    )

    # for model config
    parser.add_argument(
        "--hidden_size", type=int, required=False, help="hidden size"
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        required=False,
        help="number of attention heads",
    )
    parser.add_argument(
        "--num_layers", type=int, required=False, help="number of hidden layers"
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        required=False,
        help="maximum sequence length",
    )
    parser.add_argument(
        "--vocab_size", type=int, required=False, help="vocabulary size"
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        required=False,
        help="intermediate size",
    )

    return parser.parse_args()


def get_model_memory_usage(args):
    # evaluate model memory usage based on distributed strategy and model setting
    raise NotImplementedError(
        "Please implement this function for memory usage estimation based on distributed strategy and model setting."
    )


if __name__ == "__main__":
    args = parse_arguments()
    print(get_model_memory_usage(args))
