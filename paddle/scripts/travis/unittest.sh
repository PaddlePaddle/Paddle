#!/bin/bash
cd `dirname $0`
source ./common.sh
env CTEST_OUTPUT_ON_FAILURE=1 make test ARGS="-j `nproc`"

