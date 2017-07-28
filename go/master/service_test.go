package master_test

import (
	"os"
	"testing"
	"time"

	"github.com/PaddlePaddle/Paddle/go/master"
	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/embed"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/stretchr/testify/assert"
)

func TestNewServiceWithEtcd(t *testing.T) {
	// setup an embed etcd server
	etcdDir, err := ioutils.TempDir("", "")
	if err != nil {
		t.Fatal(err)
	}
	cfg := embed.NewConfig()
	cfg.Dir = etcdDir
	e, err := embed.StartEtcd(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		e.Close()
		if err := os.RemoveAll(etcdDir); err != nil {
			t.Fatal(err)
		}
	}()
	select {
	case <-e.Server.ReadyNotify():
		t.Log("Server is ready!")
	case <-time.After(60 * time.Second):
		e.Server.Stop() // trigger a shutdown
		t.Fatal("Server took too long to start!")
	}

	ep := []string{"127.0.0.1:2379"}
	masterAddr := "127.0.0.1:3306"
	store, err := master.NewEtcdClient(ep, masterAddr, master.DefaultLockPath, master.DefaultAddrPath, master.DefaultStatePath, 30)
	if err != nil {
		t.Fatal(err)
	}

	_, err = master.NewService(store, 10, 10, 3)
	if err != nil {
		t.Fatal(err)
	}
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   ep,
		DialTimeout: 3 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	v, err := master.GetKey(cli, master.DefaultAddrPath, 3*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if err := cli.Close(); err != nil {
		t.Fatal(err)
	}
	// test master process registry itself into etcd server.
	assert.Equal(t, masterAddr, v, "master process should registry itself into etcd server.")
}
