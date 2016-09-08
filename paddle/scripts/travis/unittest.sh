#!/bin/bash
set -e
cd `dirname $0`
cd ../../../build
env CTEST_OUTPUT_ON_FAILURE=1 make test ARGS="-j `nproc`"

