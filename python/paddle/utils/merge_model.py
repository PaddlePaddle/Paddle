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

import gzip
import struct
import os

from paddle.trainer_config_helpers.layers import LayerOutput
from paddle.v2.parameters import Parameters
from paddle.proto import ModelConfig_pb2
from paddle.v2.topology import Topology

def merge_model(net_out, param_file, output_file):
    '''Integrate the model config and model parameters into one file.
    
    The model configuration file describes the model structure which
    ends with .py. The parameters file stores the parameters of the model
    which ends with .tar.gz.
    
    @param  net_out       the output layer of the network 
    @param  param_file     path of the model parameters file(a gzip file).
    @param  output_file   path of the merged file which will be generated
    
    Usage:

        from paddle.util.merge_model import merge_model
        # import your network configuration
        from mobilenet import mobile_net
        
        net_out = mobile_net(3*224*224, 102)
        param_file = YOUR_MODEL_PARAM_PATH
        output_file = OUTPUT_MERGED_FILE_PATH
        
        merge_model(net_out, param_file, output_file)

    '''

    assert isinstance(net_out, LayerOutput), \
            "The net_out should be the output of the network"
    assert os.path.exists(param_file), \
            "The model parameters file %s does not exists " % (param_file)

    model_proto = Topology(net_out).proto()
    assert isinstance(model_proto, ModelConfig_pb2.ModelConfig)

    with gzip.open(param_file) as f: 
        params = Parameters.from_tar(f) 

    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, 'w') as f:
        param_names = [param.name for param in model_proto.parameters]
        conf_str = model_proto.SerializeToString()
        f.write(struct.pack('q', len(conf_str)))
        f.write(conf_str)
        for pname in param_names:
            params.serialize(pname, f)

    print 'Generate  %s  success!' % (output_file)
