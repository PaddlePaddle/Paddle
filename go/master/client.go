package master

import (
	"context"
	"os"
	"strings"
	"time"

	"github.com/PaddlePaddle/Paddle/go/connection"
	"github.com/PaddlePaddle/recordio"
	"github.com/coreos/etcd/clientv3"
	log "github.com/sirupsen/logrus"
)

const masterAddrPath = "/master"

// Addresser provide the address of the master server.
type Addresser interface {
	Address() string
}

// Client is the client of the master server.
type Client struct {
	conn *connection.Conn
	ch   chan []byte
}

// MasterAddresser provide master address
type MasterAddresser string

// Address return the address
func (m MasterAddresser) Address() string {
	return string(m)
}

// NewClient creates a new Client.
//
// bufSize is the record buffer size. NextRecord will read from this
// buffer.
func NewClient(addr Addresser, bufSize int) *Client {
	c := &Client{}
	c.conn = connection.New()
	c.ch = make(chan []byte, bufSize)
	go c.monitorMaster(addr)
	go c.getRecords()
	return c
}

// NewEtcdClient create a new master client by etcd
//
// etcdEndpoints is the endpoints for etcd, it's separated by "," such as
// "172.0.1.0:2379,172.0.1.1:2379"
// bufSize is the record buffer size. NextRecord will read from this buffer.
func NewEtcdClient(etcdEndpoints string, etcdTimeout int, bufSize int) *Client {
	timeout := time.Second * time.Duration(etcdTimeout)
	ep := strings.Split(etcdEndpoints, ",")
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   ep,
		DialTimeout: timeout,
	})
	if err != nil {
		log.Errorf("Init etcd connection failed: %v", err)
		panic(err)
	}
	log.Debugf("Connected to etcd: %s\n", etcdEndpoints)
	for {

		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		resp, err := cli.Get(ctx, masterAddrPath)
		cancel()
		if err != nil {
			log.Errorf("Fetch master addr failed, %v\n", err)
			time.Sleep(timeout)
			continue
		}
		kvs := resp.Kvs
		if len(kvs) == 0 {
			log.Infoln("Waiting for master be ready ...\n")
			time.Sleep(timeout)
			continue
		}

		mAddr := kvs[0].Value
		log.Debugf("Fetched master address: %s\n", mAddr)
		return NewClient(MasterAddresser(mAddr), bufSize)
	}
}

func (c *Client) getRecords() {
	for {
		t, err := c.getTask()
		if err != nil {
			// TODO(helin): wait before move on with next
			// getTask call.
			log.Errorln(err)
			continue
		}

		for _, chunk := range t.Chunks {
			f, err := os.Open(chunk.Path)
			if err != nil {
				log.Errorln(err)
				continue
			}

			s := recordio.NewRangeScanner(f, &chunk.Index, -1, -1)
			for s.Scan() {
				c.ch <- s.Record()
			}

			if s.Err() != nil {
				log.Errorln(err, chunk.Path)
			}

			err = f.Close()
			if err != nil {
				log.Errorln(err)
			}
		}

		// We treat a task as finished whenever the last data
		// instance of the task is read. This is not exactly
		// correct, but a reasonable approximation.
		c.taskFinished(t.ID)
	}
}

func (c *Client) monitorMaster(addr Addresser) {
	lastMaster := ""
	monitor := func() {
		// get the lastest address of the master server,
		// connect to the new address once address changed.
		curMaster := addr.Address()
		if curMaster != lastMaster {
			if curMaster == "" {
				err := c.conn.Close()
				if err != nil {
					log.Errorln(err)
				}
			} else {
				err := c.conn.Connect(curMaster)
				if err != nil {
					log.Errorln(err)

					// connect to addr failed, set
					// to last known addr in order
					// to retry next time.
					curMaster = lastMaster
				}

			}
		}

		lastMaster = curMaster
	}

	monitor()
	ticker := time.NewTicker(10 * time.Second)
	for _ = range ticker.C {
		monitor()
	}
}

// SetDataset set dataset for the master server to dispatch.
//
// SetDataset can be call multiple times from different nodes. But
// only the first call will be honored.
func (c *Client) SetDataset(globPaths []string) error {
	return c.conn.Call("Service.SetDataset", globPaths, nil)
}

// getTask gets a new task from the master server.
func (c *Client) getTask() (Task, error) {
	var t Task
	err := c.conn.Call("Service.GetTask", 0, &t)
	return t, err
}

// TaskFinished tells the master server a task is finished.
func (c *Client) taskFinished(taskID int) error {
	return c.conn.Call("Service.TaskFinished", taskID, nil)
}

// NextRecord returns next record in the dataset.
//
// NextRecord will block until the next record is available. It is
// thread-safe.
func (c *Client) NextRecord() []byte {
	return <-c.ch
}
