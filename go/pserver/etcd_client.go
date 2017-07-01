package pserver

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"time"

	"github.com/PaddlePaddle/Paddle/go/utils/networkhelper"
	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/clientv3/concurrency"
	log "github.com/sirupsen/logrus"
)

// EtcdClient is the etcd client that the pserver uses for fault
// tolerance, service registry and coordination.
type EtcdClient struct {
	numPservers   int
	etcdEndpoints string
	etcdClient    *clientv3.Client
	// etcdTimeout is also used as retry intervals.
	etcdTimeout time.Duration
	// FIXME: ensure GetExternalIP gets the correct ip for trainers to connect.
	externalIP string
	// desired number of pservers in the job.
	// assume desired will not change during one training job.
	desired int
}

// NewEtcdClient creates an EtcdClient
func NewEtcdClient(endpoints string, numPservers int, timeout time.Duration) *EtcdClient {
	return &EtcdClient{
		etcdTimeout:   timeout,
		numPservers:   numPservers,
		etcdEndpoints: endpoints,
	}
}

// Register registers the pserver on etcd
//
// Register returns the index of the current pserver.
func (e *EtcdClient) Register() (int, error) {

	var err error
	e.externalIP, err = networkhelper.GetExternalIP()
	if err != nil {
		return 0, err
	}

	// initialize connection to etcd.
	ep := strings.Split(e.etcdEndpoints, ",")
	for {
		cli, err := clientv3.New(clientv3.Config{
			Endpoints:   ep,
			DialTimeout: e.etcdTimeout,
		})
		if err != nil {
			log.Errorf("connect to etcd error: %v", err)
			time.Sleep(e.etcdTimeout)
			continue
		}
		e.etcdClient = cli
		log.Debugf("inited client to %s", e.etcdEndpoints)
		break
	}
	// init /ps_desired using transaction, for multiple pservers may want to write
	// it at the same time.
	for {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		_, err := e.initDesiredPsercers(ctx, e.numPservers)
		cancel()
		if err != nil {
			log.Warn(err)
			time.Sleep(e.etcdTimeout)
			continue
		}
		break
	}
	// TODO: when implementing extending or reducing pservers, /ps_desired is
	// changed, then we need to watch /ps_desired node for events. For now, just
	// write once when init and read from it.
	// wait and set s.desired init value
	for {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		resp, err := e.etcdClient.Get(ctx, PsDesired)
		cancel()
		if err != nil {
			log.Errorf("getting %s error: %v", PsDesired, err)
			time.Sleep(e.etcdTimeout)
			continue
		}
		if len(resp.Kvs) != 0 {
			e.desired, err = strconv.Atoi(string(resp.Kvs[0].Value))
			if err != nil {
				log.Errorf("value of %s invalid %v\n", PsDesired, err)
				time.Sleep(e.etcdTimeout)
				// NOTE: wait util ps_desired value change
				continue
			}
			break
		}
	}

	var pserverIdx int
	// try register pserver node on etcd
	for {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		var err error
		pserverIdx, err = e.registerPserverEtcd(ctx)
		cancel()
		if err != nil {
			log.Warn(err)
			time.Sleep(e.etcdTimeout)
			continue
		}
		break
	}

	return pserverIdx, nil
}

func (e *EtcdClient) initDesiredPsercers(ctx context.Context, numPservers int) (*clientv3.TxnResponse, error) {
	return concurrency.NewSTM(e.etcdClient, func(c concurrency.STM) error {
		dsStr := c.Get(PsDesired)
		if dsStr == "" {
			c.Put(PsDesired, strconv.Itoa(numPservers))
		}
		return nil
	}, concurrency.WithAbortContext(ctx), concurrency.WithIsolation(concurrency.RepeatableReads))
}

// registerPserverEtcd registers pserver node on etcd using transaction.
func (e *EtcdClient) registerPserverEtcd(ctx context.Context) (int, error) {
	var idx int
	_, err := concurrency.NewSTM(e.etcdClient, func(c concurrency.STM) error {
		registered := false
		for i := 0; i < e.desired; i++ {
			psKey := "/ps/" + strconv.Itoa(i)
			log.Debugf("checking %s", psKey)
			ps := c.Get(psKey)
			log.Debugf("got value (%s) for key: %s", ps, psKey)

			if ps == "" {
				resp, err := e.etcdClient.Grant(context.TODO(), 5)
				if err != nil {
					log.Fatal(err)
				}
				// find the first id and write info
				c.Put(psKey, e.externalIP, clientv3.WithLease(resp.ID))
				log.Debugf("set pserver node %s with value %s", psKey, e.externalIP)
				ch, kaerr := e.etcdClient.KeepAlive(context.TODO(), resp.ID)
				if kaerr != nil {
					log.Errorf("keepalive etcd node error: %v", kaerr)
					return kaerr
				}

				// Eat the keep alive message so etcd
				// will not expire the lease.
				go func(ch <-chan *clientv3.LeaseKeepAliveResponse) {
					ka := <-ch
					log.Debugf("keepalive: %d\n", ka.TTL)
				}(ch)
				log.Debug("register finished")
				idx = i
				registered = true
				break
			}
		}
		if registered == true {
			return nil
		}
		return errors.New("not registerd, may due to already have enough pservers")
	}, concurrency.WithAbortContext(ctx), concurrency.WithIsolation(concurrency.RepeatableReads))

	if err != nil {
		return 0, err
	}

	return idx, nil
}
