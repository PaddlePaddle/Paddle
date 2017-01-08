# Copyright (c) 2016 PaddlePaddle Authors, Inc. All Rights Reserved
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

res = []
with open('./rank-00000') as f:
    for line in f:
        pred = map(int, line.strip('\r\n;').split(";"))
        #raw prediction range from 0 to 3
        res.append([i + 1 for i in pred])

file_name = open('./data/pred.list').read().strip('\r\n')

FORECASTING_NUM = 24
header = [
    'id',
    '201604200805',
    '201604200810',
    '201604200815',
    '201604200820',
    '201604200825',
    '201604200830',
    '201604200835',
    '201604200840',
    '201604200845',
    '201604200850',
    '201604200855',
    '201604200900',
    '201604200905',
    '201604200910',
    '201604200915',
    '201604200920',
    '201604200925',
    '201604200930',
    '201604200935',
    '201604200940',
    '201604200945',
    '201604200950',
    '201604200955',
    '201604201000',
]
###################
## To CSV format ##
###################
with open(file_name) as f:
    f.next()
    print ','.join(header)
    for row_num, line in enumerate(f):
        fields = line.rstrip('\r\n').split(',')
        linkid = fields[0]
        print linkid + ',' + ','.join(map(str, res[row_num]))
