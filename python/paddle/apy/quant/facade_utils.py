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

import ap


def tuple_identity_infer_symbolic(infer_ctx, inputs, attrs):
    return inputs


def tuple_identity_infer_meta(inputs, attrs, mut_outputs):
    def copy_meta(i):
        mut_outputs[i].dims = inputs[i].dims
        mut_outputs[i].dtype = inputs[i].dtype

    ap.map(copy_meta, range(len(inputs)))


def quant_infer_symbolic(infer_ctx, inputs, attrs):
    return [inputs[0], inputs[0]]


def quant_infer_meta(inputs, attrs, mut_outputs):
    mut_outputs[0].dims = inputs[0].dims
    mut_outputs[0].dtype = inputs[0].dtype
    mut_outputs[1].dims = inputs[0].dims
    mut_outputs[1].dtype = inputs[0].dtype
