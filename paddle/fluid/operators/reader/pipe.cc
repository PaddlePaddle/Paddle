// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>

#include "paddle/fluid/operators/reader/pipe.h"
#include "paddle/fluid/platform/enforce.h"

using paddle::operators::reader::Pipe;
using paddle::operators::reader::ReadPipe;
using paddle::operators::reader::WritePipe;

typedef enum {
  PIPE_SUCCESS,
  PIPE_CLOSED_ERROR,
  PARTIAL_WRITE_ERROR,
  WAIT_TIMEOUT_ERROR,
  SYSTEM_ERROR,
} pipe_error_t;

pipe_error_t pipe_init(int *read_fd, int *write_fd);
pipe_error_t pipe_read(int read_fd, uint8_t *buffer, size_t size,
                       size_t *bytes_read);
pipe_error_t pipe_write(int write_fd, const uint8_t *buffer, size_t size,
                        size_t *bytes_write);
void pipe_close(int fd);

pipe_error_t pipe_init(int *read_fd, int *write_fd) {
  int pipe_fd[2];
  int result = pipe(pipe_fd);
  fcntl(pipe_fd[0], F_SETFD, FD_CLOEXEC);
  fcntl(pipe_fd[1], F_SETFD, FD_CLOEXEC);

  if (result == -1) {
    return SYSTEM_ERROR;
  }

  *read_fd = pipe_fd[0];
  *write_fd = pipe_fd[1];

  return PIPE_SUCCESS;
}

pipe_error_t pipe_read(int read_fd, uint8_t *buffer, size_t size,
                       size_t *bytes_read) {
  *bytes_read = 0;
  ssize_t error = read(read_fd, buffer, size);
  if (error == 0) {
    return PIPE_CLOSED_ERROR;
  } else if (error == -1) {
    return SYSTEM_ERROR;
  }
  *bytes_read = (size_t)error;
  return PIPE_SUCCESS;
}

pipe_error_t pipe_write(int write_fd, const uint8_t *buffer, size_t size,
                        size_t *bytes_write) {
  *bytes_write = 0;
  ssize_t error = write(write_fd, buffer, size);
  if (error == -1) {
    switch (errno) {
      case EPIPE:
        return PIPE_CLOSED_ERROR;
      default:
        return SYSTEM_ERROR;
    }
  }
  *bytes_write = (size_t)error;
  if (*bytes_write != size) {
    return PARTIAL_WRITE_ERROR;
  }
  return PIPE_SUCCESS;
}

void pipe_close(int fd) { close(fd); }

Pipe::Pipe(int file_descriptor) : file_descriptor_(file_descriptor) {}

Pipe::~Pipe() { pipe_close(file_descriptor_); }

std::array<int, 2> Pipe::Create() {
  int r, w;
  pipe_error_t error = pipe_init(&r, &w);
  PADDLE_ENFORCE_EQ(error, PIPE_SUCCESS, "Pipe initialization error");
  return std::array<int, 2>({r, w});
}

void ReadPipe::read(uint8_t *buffer, std::size_t size) {
  std::size_t read_bytes;
  std::size_t remain_bytes = size;
  while (remain_bytes > 0) {
    pipe_error_t error =
        pipe_read(file_descriptor_, buffer, remain_bytes, &read_bytes);
    remain_bytes -= read_bytes;
    buffer += read_bytes;
    PADDLE_ENFORCE_EQ(error, PIPE_SUCCESS, "ReadPipe read error");
  }
}

void WritePipe::write(const uint8_t *buffer, std::size_t size) {
  std::size_t write_bytes;
  pipe_error_t error = pipe_write(file_descriptor_, buffer, size, &write_bytes);
  PADDLE_ENFORCE_EQ(error, PIPE_SUCCESS, "WritePipe write error");
}
