## Table VarType Design

A Table Variable is a generally a `map` used to support distributed lookup table. In distributed training, a big embedding table will be divided into multiple `Table` Variables and initialized on each parameter servers. 

The number of the divided tables is equal to the number of parameter server, which will be stored in Table as `shard_num`.

Each Table also has a `shard_id`, a key with the same `shard_id` will be sent to this parameter server.

The way to get the `shard_id` can be:

```cpp
int shard_id(const KEY& key) {
    return hash(key) % shard_num;
}
```

Optimization op takes `Table` as parameter input, then apply the gradient to it.

Take `sgd` as an example: 

```
sgd(parameter<Table>, grad<SelectedRows>) -> parameter<Table>
```

`Table` in protobuf:

```proto
message VarType {
  enum Type {
    // Pod Types
    BOOL = 0;
    ...

    // Other types that may need additional descriptions
    LOD_TENSOR = 7;
    TABLE = 20;
  }

  required Type type = 1;

  message LoDTensorDesc {
    required TensorDesc tensor = 1;
    optional int32 lod_level = 2 [ default = 0 ];
  }
  optional LoDTensorDesc lod_tensor = 3;
  
  message TableDesc {
    required int32 shard_num = 1; // the total number of shard
    required int32 shard_id = 2; // the id of this shard
  }
  optional TableDesc table = 3;
}
```

`Table` in CPP:

```cpp
template<class KEY, class VALUE>
class Table {
 public:
  Table(int shard_num, int shard_id) :
          shard_num_(shard_num), shard_id_(shard_id) {
    map_.reset(new std::unordered_map<KEY, VALUE>());
  }

  std::vector<VALUE> get(const std::vector<KEY>& keys) const;
  const VALUE& get(const KEY& key) const;

  void update(const KEY& key, const VALUE& value);
  void update(const std::vector<KEY>& keys, const std::vector<VALUE>& values);

 private:
  int shard_num_;
  int shard_id_;
  std::unique_ptr<std::unordered_map<KEY, VALUE>> map_;
};
```
table should use buddy allocator to allocate memory.
