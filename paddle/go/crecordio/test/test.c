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
  write_recordio(w, "hello", 6);
  write_recordio(w, "hi", 3);
  release_recordio(w);

  w = create_recordio_writer("/tmp/test_recordio_1");
  write_recordio(w, "dog", 4);
  write_recordio(w, "cat", 4);
  release_recordio(w);

  reader r = create_recordio_reader("/tmp/test_recordio_*");
  int size;
  unsigned char* item = read_next_item(r, &size);
  if (strcmp(item, "hello") || size != 6) {
    fail();
  }

  free(item);

  item = read_next_item(r, &size);
  if (strcmp(item, "hi") || size != 3) {
    fail();
  }
  free(item);

  item = read_next_item(r, &size);
  if (strcmp(item, "dog") || size != 4) {
    fail();
  }
  free(item);

  item = read_next_item(r, &size);
  if (strcmp(item, "cat") || size != 4) {
    fail();
  }
  free(item);

  item = read_next_item(r, &size);
  if (item != NULL || size != -1) {
    fail();
  }

  release_recordio_reader(r);
}
