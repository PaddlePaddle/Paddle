#!/bin/bash
cd `dirname $0`
source ./common.sh
sudo make install
sudo paddle version
