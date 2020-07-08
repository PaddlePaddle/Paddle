#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define the functions to calculate metric in this directory 
__all__ = [
    'Accuracy', 'Auc', 'ChunkEvaluator', 'CompositeMetric', 'DetectionMAP',
    'EditDistance', 'Precision', 'Recall', 'accuracy', 'auc', 'chunk_eval',
    'cos_sim', 'mean_iou'
]



from ..fluid.metrics import Accuracy, Auc, ChunkEvaluator, CompositeMetric, DetectionMAP, EditDistance, \
        Precision, Recall

from ..fluid.layers.metric_op import accuracy, auc
from ..fluid.layers.nn import chunk_eval, cos_sim, mean_iou
