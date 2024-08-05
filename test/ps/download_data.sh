# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

wget --no-check-certificate https://fleet.bj.bcebos.com/ctr_data.tar.gz
tar -zxvf ctr_data.tar.gz
mv ./raw_data ./train_data_full
mkdir train_data && cd train_data
cp ../train_data_full/part-0 ../train_data_full/part-1 ./ && cd ..
mv ./test_data ./test_data_full
mkdir test_data && cd test_data
cp ../test_data_full/part-220 ./  && cd ..
echo "Complete data download."
echo "Full Train data stored in ./train_data_full "
echo "Full Test data stored in ./test_data_full "
echo "Rapid Verification train data stored in ./train_data "
echo "Rapid Verification test data stored in ./test_data "
