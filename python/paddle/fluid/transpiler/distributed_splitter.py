#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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


def hash_name(varlist, pserver_endpoints):
    """
    hash variable names to several endpoints.

    Args:
        varlist(list): a list of Variables

    Returns(dict): a map of pserver endpoint -> varname
    """

    def _hash_block(block_str, total):
        return hash(block_str) % total

    eplist = []
    for var in varlist:
        server_id = _hash_block(var.name(), len(pserver_endpoints))
        server_for_param = pserver_endpoints[server_id]
        eplist.append(server_for_param)
    return eplist


def round_robin(varlist, pserver_endpoints):
    """
    Distribute variables to several endpoints.
    Args:
        varlist(list): a list of variables
        pserver_endpoints(list): a list of pserver endpoints

    Returns(list[int]): the endpoint for each variable
    """
    assert (len(varlist) >= len(pserver_endpoints))

    eplist = []
    pserver_idx = 0
    for var in varlist:
        server_for_param = pserver_endpoints[pserver_idx]
        eplist.append(server_for_param)

        pserver_idx += 1
        if pserver_idx >= len(pserver_endpoints):
            pserver_idx = 0
    return eplist
