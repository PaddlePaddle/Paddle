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
import sys

# kinetics-400_train.csv should be down loaded first and set as sys.argv[1]
# sys.argv[2] can be set as kinetics400_label.txt
# python generate_label.py kinetics-400_train.csv kinetics400_label.txt

num_classes = 400

fname = sys.argv[1]
outname = sys.argv[2]
fl = open(fname).readlines()
fl = fl[1:]
outf = open(outname, 'w')

label_list = []
for line in fl:
    label = line.strip().split(',')[0].strip('"')
    if label in label_list:
        continue
    else:
        label_list.append(label)

assert len(label_list
           ) == num_classes, "there should be {} labels in list, but ".format(
               num_classes, len(label_list))

label_list.sort()
for i in range(num_classes):
    outf.write('{} {}'.format(label_list[i], i) + '\n')

outf.close()
