#include <thread>
#include <iostream>
#include <memory>
#include <unistd.h>

using namespace std;

class TestClass {
 public:
  TestClass() {
    cout << "Constructor" << endl;
  }
  ~TestClass() {
    cout << "Desctructor" << endl;
  }
};

void* GetAddr() {
  thread_local TestClass obj;
  return &obj;
}

void Func() {
  cout << GetAddr() << endl;
}

int main() {
  int a;
  auto deleter = [](int* a){ cout << "dummy deleter" << endl;};
  std::unique_ptr<int, decltype(deleter)>(&a, deleter);
  std::thread t1(Func);
  std::thread t2(Func);
  sleep(10);
}
