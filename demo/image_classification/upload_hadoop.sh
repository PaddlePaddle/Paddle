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
set -e
hadoop fs -Dhadoop.job.ugi=paddle_demo,paddle_demo -put data/cifar-out/batches/train_batch_* /app/idl/idl-dl/paddle/demo/image_classification/train/
hadoop fs -Dhadoop.job.ugi=paddle_demo,paddle_demo -put data/cifar-out/batches/test_batch_* /app/idl/idl-dl/paddle/demo/image_classification/test/
hadoop fs -Dhadoop.job.ugi=paddle_demo,paddle_demo -put data/cifar-out/batches/batches.meta /app/idl/idl-dl/paddle/demo/image_classification/train_meta
hadoop fs -Dhadoop.job.ugi=paddle_demo,paddle_demo -put data/cifar-out/batches/batches.meta /app/idl/idl-dl/paddle/demo/image_classification/test_meta
