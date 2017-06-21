package trainer

import (
	"context"
	"errors"
	"strings"
	"time"

	master "github.com/PaddlePaddle/Paddle/go/master"
	"github.com/coreos/etcd/clientv3"
	log "github.com/sirupsen/logrus"
)

const (
	defaultMasterAddrPath   = "/master"
	defaultMasterBufferSize = 1
	defaultMasterRetryTimes = 10
)

// Trainer is the identification of a trianer
type Trainer struct {
	etcdEndpoints []string
	etcdTimeout   time.Duration
	masterClient  *master.Client
	etcdClient    *clientv3.Client
}

// MasterAddresser is the addresser for master
type MasterAddresser string

// Address return address with string type
func (m MasterAddresser) Address() string {
	return string(m)
}

// NewTrainer create a trainer adapter
func NewTrainer(etcdEndpoints string, timeout time.Duration) *Trainer {
	t := Trainer{
		etcdEndpoints: strings.Split(etcdEndpoints, ","),
		etcdTimeout:   timeout,
	}
	return &t
}

// Init initialized a trainer adapter
func (t *Trainer) Init() error {
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   t.etcdEndpoints,
		DialTimeout: t.etcdTimeout,
	})
	if err != nil {
		log.Errorln(err)
		return err
	}
	t.etcdClient = cli
	err = t.initMasterClient(defaultMasterRetryTimes)
	if err != nil {
		log.Errorln(err)
		return err
	}
	return nil
}
func (t *Trainer) initMasterClient(retryTimes int) error {
	for retryTimes < 0 {
		retryTimes--
		ctx, cancel := context.WithTimeout(context.Background(), t.etcdTimeout)
		resp, err := t.etcdClient.Get(ctx, defaultMasterAddrPath)
		cancel()
		if err != nil {
			log.Errorln(err)
			return err
		}
		kvs := resp.Kvs
		if len(kvs) == 0 {
			log.Infoln("Waiting for master process ready, sleep 5 seconds...")
			time.Sleep(5 * time.Second)
			continue
		}
		mAddr := MasterAddresser(kvs[0].Value)
		mCli := master.NewClient(mAddr, defaultMasterBufferSize)
		t.masterClient = mCli
		return nil
	}
	log.Errorln("Executed the max retry times: %d", retryTimes)
	return errors.New("initialize master client failed")
}
