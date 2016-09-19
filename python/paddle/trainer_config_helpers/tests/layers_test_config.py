# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
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

from paddle.trainer_config_helpers import *

num_classes = 5

x = data_layer(name="input1", size=3)
y = data_layer(name="input2", size=5)

z = out_prod_layer(input1=x, input2=y)

x1 = fc_layer(input=x, size=5)
y1 = fc_layer(input=y, size=5)
y2 = fc_layer(input=y, size=15)

cos1 = cos_sim(a=x1, b=y1)
cos3 = cos_sim(a=x1, b=y2, size=3)

linear_comb = linear_comb_layer(weights=x1, vectors=y2, size=3)

out = fc_layer(input=[cos1, cos3, linear_comb, z],
               size=num_classes,
               act=SoftmaxActivation())

print_layer(input=[out])

outputs(classification_cost(out, data_layer(name="label", size=num_classes)))

dotmul = mixed_layer(input=[dotmul_operator(x=x1, y=y1),
	                        dotmul_projection(input=y1)])

# for ctc
tmp = fc_layer(input=[x1, dotmul],
               size=num_classes + 1,
               act=SoftmaxActivation())
ctc = ctc_layer(input=tmp,
                label=y,
                size=num_classes + 1)
ctc_eval = ctc_error_evaluator(input=tmp, label=y)

settings(
    batch_size=10,
    learning_rate=2e-3,
    learning_method=AdamOptimizer(),
    regularization=L2Regularization(8e-4),
    gradient_clipping_threshold=25
)
