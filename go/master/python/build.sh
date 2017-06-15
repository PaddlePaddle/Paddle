#!/bin/bash

go build -buildmode=c-shared ../c && rm c.h && mv c paddle_master/libmaster.so
pip wheel .
