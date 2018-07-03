/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef HL_GPU_H_
#define HL_GPU_H_

#include "hl_aggregate.h"
#include "hl_base.h"
#include "hl_cnn.h"
#include "hl_cuda.h"
#include "hl_cuda_cublas.h"
#include "hl_cuda_cudnn.h"
#include "hl_lstm.h"
#include "hl_matrix.h"
#include "hl_sequence.h"
#include "hl_sparse.h"
#ifndef PADDLE_MOBILE_INFERENCE
#include "hl_warpctc_wrap.h"
#endif

#ifdef HPPL_STUB_FUNC
#include "stub/hl_aggregate_stub.h"
#include "stub/hl_cnn_stub.h"
#include "stub/hl_cuda_cublas_stub.h"
#include "stub/hl_cuda_cudnn_stub.h"
#include "stub/hl_cuda_stub.h"
#include "stub/hl_lstm_stub.h"
#include "stub/hl_matrix_stub.h"
#include "stub/hl_sequence_stub.h"
#include "stub/hl_sparse_stub.h"
#endif

#endif /* HL_GPU_H_ */
