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

set -x
for ((port=12340;port<=12360;port++))
do
    port_used_num=`netstat -a |grep $port|wc -l`
    if [ $port_used_num -eq 0 ]
    then
        echo $port;
        pserver/test/test_ProtoServer --port=$port 
        if [ $? -eq 0 ]
           then
               exit 0
           else
               echo "test_ProtoServer run wrong"
       	       exit 1
        fi
fi
done
echo "test_ProtoServer port not found"
exit 1
