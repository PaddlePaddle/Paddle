#!/bin/bash
set -e
docker build -t build_paddle_deb .
rm -rf dist
mkdir -p dist
docker run -v$PWD/dist:/root/dist -v $PWD/../../../..:/root/paddle --name tmp_build_deb_container build_paddle_deb
docker rm tmp_build_deb_container
docker rmi build_paddle_deb
