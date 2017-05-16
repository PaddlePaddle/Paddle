cmake_minimum_required(VERSION 3.0)

ExternalGoProject_Add(pserver github.com/PaddlePaddle/Paddle/paddle/go/pserver)
add_go_library(client STATIC pserver)
add_subdirectory(test)
