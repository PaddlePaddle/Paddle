package client_test

import (
	"io/ioutil"
	"os"
	"sync"
	"testing"

	"github.com/PaddlePaddle/Paddle/go/pserver/client"
	"github.com/coreos/etcd/embed"
)

func TestSelector(t *testing.T) {
	etcdDir, err := ioutil.TempDir("", "")
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

	<-e.Server.ReadyNotify()

	var mu sync.Mutex
	selectedCount := 0
	var wg sync.WaitGroup
	selectAndDone := func(c *client.Etcd) {
		defer wg.Done()

		selected, err := c.Select()
		if err != nil {
			panic(err)
		}

		if selected {
			mu.Lock()
			selectedCount++
			mu.Unlock()
			err = c.Done()
			if err != nil {
				t.Fatal(err)
			}
		}
	}

	c0 := client.NewEtcd("127.0.0.1:2379")
	c1 := client.NewEtcd("127.0.0.1:2379")
	c2 := client.NewEtcd("127.0.0.1:2379")
	c3 := client.NewEtcd("127.0.0.1:2379")
	wg.Add(3)
	go selectAndDone(c0)
	go selectAndDone(c1)
	go selectAndDone(c2)
	wg.Wait()

	// simulate trainer crashed and restarted after the
	// initialization process.
	wg.Add(1)
	go selectAndDone(c3)
	wg.Wait()

	mu.Lock()
	if selectedCount != 1 {
		t.Fatal("selected count wrong:", selectedCount)
	}
	mu.Unlock()

	err = c0.Close()
	if err != nil {
		t.Fatal(err)
	}

	err = c1.Close()
	if err != nil {
		t.Fatal(err)
	}

	err = c2.Close()
	if err != nil {
		t.Fatal(err)
	}

	err = c3.Close()
	if err != nil {
		t.Fatal(err)
	}
}
