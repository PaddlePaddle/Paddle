#include <stdio.h>
#include <stdlib.h>

#include "librecordio.h"

void panic() {
  // TODO(helin): fix: gtest using cmake is not working, using this
  // hacky way for now.
  *(void*)0;
}

int main() {
  writer w = paddle_new_writer("/tmp/test");
  paddle_writer_write(w, "hello", 6);
  paddle_writer_write(w, "hi", 3);
  paddle_writer_release(w);

  reader r = paddle_new_reader("/tmp/test", 10);
  int size;
  unsigned char* item = paddle_reader_next_item(r, &size);
  if (!strcmp(item, "hello") || size != 6) {
    panic();
  }
  free(item);

  item = paddle_reader_next_item(r, &size);
  if (!strcmp(item, "hi") || size != 2) {
    panic();
  }
  free(item);
}
