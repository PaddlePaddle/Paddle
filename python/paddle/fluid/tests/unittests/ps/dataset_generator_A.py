#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid.incubate.data_generator as dg

cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
hash_dim_ = 1000001
continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)


class CriteoDataset(dg.MultiSlotDataGenerator):
<<<<<<< HEAD

=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    def generate_sample(self, line):
        """
        Read the data line by line and process it as a dictionary
        """

        def reader():
            """
            This function needs to be implemented by the user, based on data format
            """
            features = line.rstrip('\n').split('\t')
            feature_name = []
            sparse_feature = []
            for idx in categorical_range_:
                sparse_feature.append(
<<<<<<< HEAD
                    [hash(str(idx) + features[idx]) % hash_dim_])
=======
                    [hash(str(idx) + features[idx]) % hash_dim_]
                )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
            for idx in categorical_range_:
                feature_name.append("C" + str(idx - 13))
            yield list(zip(feature_name, sparse_feature))

        return reader


d = CriteoDataset()
d.run_from_stdin()
