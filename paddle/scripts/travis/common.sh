#!/bin/bash
set -e
mkdir -p ../../../build
cd ../../../build
mkdir -p $HOME/third_party
EXTRA_CMAKE_OPTS="-DTHIRD_PARTY_PATH=${HOME}/third_party"
