#include <stdio.h>
#include <stdlib.h>

#include "librecordio.h"

void fail() {
  // TODO(helin): fix: gtest using cmake is not working, using this
  // hacky way for now.
  printf("test failed.\n");
  exit(-1);
}

int main() {
  writer w = create_recordio_writer("/tmp/test_recordio_0");
  recordio_write(w, "hello", 6);
  recordio_write(w, "hi", 3);
  release_recordio_writer(w);

  w = create_recordio_writer("/tmp/test_recordio_1");
  recordio_write(w, "dog", 4);
  recordio_write(w, "cat", 4);
  release_recordio_writer(w);

  reader r = create_recordio_reader("/tmp/test_recordio_*");
  unsigned char* item = NULL;
  int size = recordio_read(r, &item);
  if (strcmp(item, "hello") || size != 6) {
    fail();
  }
  free(item);

  size = recordio_read(r, &item);
  if (strcmp(item, "hi") || size != 3) {
    fail();
  }
  free(item);

  size = recordio_read(r, &item);
  if (strcmp(item, "dog") || size != 4) {
    fail();
  }
  free(item);

  size = recordio_read(r, &item);
  if (strcmp(item, "cat") || size != 4) {
    fail();
  }
  free(item);

  size = recordio_read(r, &item);
  if (size != -1) {
    fail();
  }

  release_recordio_reader(r);
}
