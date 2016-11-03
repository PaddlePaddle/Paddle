#!/bin/bash
set -e
docker build . -t paddle_build_doc
docker run --rm -v $PWD/../../../../:/paddle -v $PWD:/output paddle_build_doc
