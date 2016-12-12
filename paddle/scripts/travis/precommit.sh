#!/bin/bash
set -e
source common.sh
cd ..
pre-commit install
pre-commit run -a
