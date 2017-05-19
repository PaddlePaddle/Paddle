package pserver

type partitioner struct {
	shardNum int
}

// partitioner partitions the parameters into shards.
func newPartitioner(shardNum int) *partitioner {
	return &partitioner{shardNum: shardNum}
}
