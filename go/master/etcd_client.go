package master

import (
	"context"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/clientv3/concurrency"
	log "github.com/sirupsen/logrus"
)

const (
	// DefaultLockPath is the default etcd master lock path.
	DefaultLockPath = "/master/lock"
	// DefaultStatePath is the default etcd key for master state.
	DefaultStatePath = "/master/state"
	// DefaultAddrPath is the default etcd key for master address.
	DefaultAddrPath = "/master/addr"
)

// EtcdClient is the etcd client that the master uses for fault
// tolerance and service registry.
type EtcdClient struct {
	lockPath  string
	statePath string
	client    *clientv3.Client
	lock      *concurrency.Mutex
}

// NewEtcdClient creates a new EtcdClient.
func NewEtcdClient(endpoints []string, addr string, lockPath, addrPath, statePath string, ttlSec int) (*EtcdClient, error) {
	log.Debugf("Connecting to etcd at %v", endpoints)
	// TODO(helin): gracefully shutdown etcd store. Becuase etcd
	// store holds a etcd lock, even though the lock will expire
	// when the lease timeout, we need to implement graceful
	// shutdown to release the lock.
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: dialTimeout,
	})
	if err != nil {
		return nil, err
	}

	sess, err := concurrency.NewSession(cli, concurrency.WithTTL(ttlSec))
	if err != nil {
		return nil, err
	}

	lock := concurrency.NewMutex(sess, lockPath)
	// It's fine for the lock to get stuck, in this case we have
	// multiple master servers running (only configured to have
	// one master running, but split-brain problem may cuase
	// multiple master servers running), and the cluster management
	// software will kill one of them.
	log.Debugf("Trying to acquire lock at %s.", lockPath)
	err = lock.Lock(context.TODO())
	if err != nil {
		return nil, err
	}
	log.Debugf("Successfully acquired lock at %s.", lockPath)

	put := clientv3.OpPut(addrPath, string(addr))
	resp, err := cli.Txn(context.Background()).If(lock.IsOwner()).Then(put).Commit()
	if err != nil {
		return nil, err
	}

	if !resp.Succeeded {
		log.Fatal("No longer owns the master lock. Exiting.")
	}

	e := &EtcdClient{
		lockPath:  lockPath,
		statePath: statePath,
		client:    cli,
		lock:      lock,
	}

	return e, nil
}

// Save saves the state into the etcd.
func (e *EtcdClient) Save(state []byte) error {
	ctx := context.TODO()
	put := clientv3.OpPut(e.statePath, string(state))
	resp, err := e.client.Txn(ctx).If(e.lock.IsOwner()).Then(put).Commit()
	if err != nil {
		return err
	}

	if !resp.Succeeded {
		log.Errorln("No longer owns the lock, trying to lock again")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		err := e.lock.Lock(ctx)
		cancel()
		if err != nil {
			// We lost the master lock and can not acquire
			// it back, it means some other master is
			// already started. We don't want cluster
			// managment system to kill the master server
			// who is holding the lock and running
			// correctly. So the most feasible solution is
			// to kill current master server. The current
			// state is not saved, but the trainer's RPC
			// call will fail, so the trainer will retry.
			log.Fatalf("Could not acquire the lock at %s: %v. Exiting.", e.lockPath, err)
		}
		log.Infof("Successfully acquired lock at %s.", e.lockPath)
		return e.Save(state)
	}

	return nil
}

// Load loads the state from etcd.
func (e *EtcdClient) Load() ([]byte, error) {
	ctx := context.TODO()
	get := clientv3.OpGet(e.statePath)

	resp, err := e.client.Txn(ctx).If(e.lock.IsOwner()).Then(get).Commit()
	if err != nil {
		return nil, err
	}

	if !resp.Succeeded {
		log.Errorln("No longer owns the lock, trying to lock and load again.")
		err = e.lock.Lock(context.Background())
		if err != nil {
			return nil, err
		}

		return e.Load()
	}

	kvs := resp.Responses[0].GetResponseRange().Kvs
	if len(kvs) == 0 {
		// No state exists
		return nil, nil
	}

	state := kvs[0].Value
	return state, nil
}

// GetKey gets the value by the specify key.
func GetKey(c *clientv3.Client, key string, timeout int) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*time.Duration(timeout))
	resp, err := c.Get(ctx, key)
	cancel()
	if err != nil {
		return "", err
	}
	kvs := resp.Kvs
	if len(kvs) == 0 {
		return "", nil
	}
	v := kvs[0].Value
	return string(v), nil
}

// WatchKey watches the specify key and send to valChan if there is some event.
func WatchKey(c *clientv3.Client, key string, valChan chan<- string) {
	rch := c.Watch(context.Background(), key)
	for wresp := range rch {
		for _, ev := range wresp.Events {
			// if received event is DELETE, the value will be an empty string
			log.Infof("received event %s, %q : %q\n", ev.Type, ev.Kv.Key, ev.Kv.Value)
			valChan <- string(ev.Kv.Value)
		}
	}
}
