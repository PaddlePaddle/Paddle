# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
import collections

from paddle.trainer_config_helpers.layers import LayerOutput
from paddle.v2.layer import parse_network
from paddle.proto import TrainerConfig_pb2

__all__ = ["dump_v2_config"]


def dump_v2_config(topology, save_path, binary=False):
    """ Dump the network topology to a specified file.

    This function is only used to dump network defined by using PaddlePaddle V2
    APIs. This function will NOT dump configurations related to PaddlePaddle
    optimizer.

    :param topology: The output layers (can be more than one layers given in a
                     Python List or Tuple) of the entire network. Using the
                     specified layers (if more than one layer is given) as root,
                     traversing back to the data layer(s), all the layers
                     connected to the specified output layers will be dumped.
                     Layers not connceted to the specified will not be dumped.
    :type topology: LayerOutput|List|Tuple
    :param save_path: The path to save the dumped network topology.
    :type save_path: str
    :param binary: Whether to dump the serialized network topology or not.
                   The default value is false. NOTE that, if you call this
                   function to generate network topology for PaddlePaddle C-API,
                   a serialized version of network topology is required. When
                   using PaddlePaddle C-API, this flag MUST be set to True.
    :type binary: bool
    """

    if isinstance(topology, LayerOutput):
        topology = [topology]
    elif isinstance(topology, collections.Sequence):
        for out_layer in topology:
            assert isinstance(out_layer, LayerOutput), (
                "The type of each element in the parameter topology "
                "should be LayerOutput.")
    else:
        raise RuntimeError("Error input type for parameter topology.")

    model_str = parse_network(topology)
    with open(save_path, "w") as fout:
        if binary:
            fout.write(model_str.SerializeToString())
        else:
            fout.write(str(model_str))
