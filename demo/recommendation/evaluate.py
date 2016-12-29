#!/usr/bin/python
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
import sys
import re
import math


def get_best_pass(log_filename):
    with open(log_filename, 'r') as f:
        text = f.read()
        pattern = re.compile('Test.*? cost=([0-9]+\.[0-9]+).*?pass-([0-9]+)',
                             re.S)
        results = re.findall(pattern, text)
        sorted_results = sorted(results, key=lambda result: float(result[0]))
        return sorted_results[0]


log_filename = sys.argv[1]
log = get_best_pass(log_filename)
predict_error = math.sqrt(float(log[0])) / 2
print 'Best pass is %s, error is %s, which means predict get error as %f' % (
    log[1], log[0], predict_error)

evaluate_pass = "output/pass-%s" % log[1]
print "evaluating from pass %s" % evaluate_pass
