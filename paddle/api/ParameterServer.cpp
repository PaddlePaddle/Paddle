//
// Created by qiaolongfei on 16/12/29.
//
#include "PaddleAPI.h"

#include "PaddleAPIPrivate.h"

ParameterServer::ParameterServer() : m(new ParameterServerPrivate()) {}

ParameterServer::~ParameterServer() { delete m; }
