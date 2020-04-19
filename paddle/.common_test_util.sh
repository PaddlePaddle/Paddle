#!/bin/bash
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

PORT_FILE=/tmp/paddle_test_ports
PORT_LOCK_FILE=/tmp/paddle_test_ports.lock

# Create flag file, all user can rw, ignore all error here
touch $PORT_FILE $PORT_LOCK_FILE 2>/dev/null
chmod a+rw $PORT_FILE $PORT_LOCK_FILE 2>/dev/null

# acquire a range of ports that not used by other runtests.sh currentlly.
# return 1 if ports is used by other, otherwise return 0.
# NOTE: the acquire_ports/release_ports is interprocess mutexed.
#
# There are two parameter of this method
# param 1: the begin of port range
# param 2: the length of port range.
# so, the port range is [param1, param1+param2)
acquire_ports(){
  (
    flock -x 200
    let "len=$1+$2"
    for((i=$1; i<$len; i++))
    do
      grep -q $i $PORT_FILE
      if [ $? -eq 0 ] ; then
        return 1 # Port already write to $PORT_FILE
      fi
    done

    for((i=$1; i<$len; i++))
    do
      echo $i >> $PORT_FILE # Write to $PORT_FILE
    done
    return 0
  )200>$PORT_LOCK_FILE
}

# release a range of ports. Mark these ports is not used by runtests.sh.
# NOTE: the acquire_ports/release_ports is interprocess mutexed.
#
# The parameter is same as acquire_ports, see acquire_ports' comments.
release_ports(){
  (
    flock -x 200
    let "len=$1+$2"
    for((i=$1; i<$len; i++))
    do
      tmp=`sed "/$i/d" $PORT_FILE`  # remove port
      echo $tmp > $PORT_FILE
    done
  )200>$PORT_LOCK_FILE
}

# use set_port  to get a random free port
# such as    set_port -p port test_fuc   to run  test_fuc --port=random
# use  -n to set_port test_fuc to get a continuous free port
# such as    set_port  -n 10 -p port  test_fuc  to get ten continuous free port to run test_fuc --port=random
set_port()
{
    num=1

    port_type="port"
    unset OPTIND
    while   getopts  "n:p:"  opt
    do
        case  "$opt"   in
            n)   echo  "get num ${OPTARG}"
                 num=${OPTARG}
                 ;;
            p)   echo  "get port_type ${OPTARG}"
                 port_type=${OPTARG}
                 ;;
        esac
    done
    shift $((OPTIND-1))
    cmd=$@
    for ((i=1;i<=10000;i++))
    do
        declare -i port=$RANDOM+10000
        port_used_total=0
        for((n=0;n<=num-1;n++))
            do
                declare -i port_check=$port+$n
                port_used_num=`netstat -a |grep $port_check|wc -l`
                declare -i port_used_total=$port_used_total+$port_used_num
            done
        if [ $port_used_total -ne 0 ]
            then
                continue
        fi
        # Lock Ports.
        acquire_ports $port $num
        if [ $? -ne 0 ]; then
            continue
        fi
        $cmd --$port_type=$port
        return_val=$?
        release_ports $port $num
        if [ $return_val -eq 0 ]; then
            return 0
        else
            echo "$cmd run wrong"
            return 1
        fi
    done

}
