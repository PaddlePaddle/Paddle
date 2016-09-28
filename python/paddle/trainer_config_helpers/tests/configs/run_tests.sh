#!/bin/bash
cd `dirname $0`
set -e
./generate_protostr.sh
md5sum -c check.md5
