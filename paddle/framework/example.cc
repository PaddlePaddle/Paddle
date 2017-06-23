#include <iostream>
#include "paddle/framework/example.pb.h"

int main() {
  paddle::framework::Something s;
  s.set_something(123.45);
  std::cout << s.something() << "\n";
  return 0;
}
