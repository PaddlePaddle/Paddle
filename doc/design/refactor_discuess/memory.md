## Workspace

```cpp
class TensorBuffer {
public:
   virtual void* get() = 0;
   virtual size_t size() const = 0;
};
struct Workspace {
  Map<string, TensorBufferPtr> allBuffers_;
}
```
所有的参数全部存放到一个WorkSpace中。申请、释放、resize `TensorBuffer`交由这个全局的WorkSpace完成。这个WorkSpace的好处是:

1. 可以在不同的拓扑结构中，共享一段内存。(share 同一个`Workspace`即可)
2. 可以实现check point机制。即将所有的buffer序列化后即完成了checkpoint。`Workspace.checkpoint()`
