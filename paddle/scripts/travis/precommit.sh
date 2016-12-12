#!/bin/bash
set -e
source common.sh
cd ..
export PATH=/usr/bin:$PATH
pre-commit install
clang-format --version
pre-commit run -a
