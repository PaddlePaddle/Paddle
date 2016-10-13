#!/bin/bash
set -e
docker build -t build_paddle_rpm .
rm -rf dist
mkdir -p dist
docker run -v$PWD/dist:/root/dist --name tmp_build_rpm_container build_paddle_rpm
docker rm tmp_build_rpm_container
docker rmi build_paddle_rpm
