package pserver

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/PaddlePaddle/Paddle/go/utils/networkhelper"
	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/clientv3/concurrency"
	log "github.com/sirupsen/logrus"
)

// ElementType is the type of elements of a Parameter.
type ElementType int

const (
	AlreadyInitialized = "pserver already initialized"
	Uninitialized      = "pserver not fully initialized"
)

// Supported element types
const (
	Int32 ElementType = iota
	UInt32
	Int64
	UInt64
	Float32
	Float64
)

// PsDesired is etcd path for store desired pserver count
const PsDesired = "/ps_desired"

// Parameter is a piece of data to sync with the parameter server.
type Parameter struct {
	Name        string
	ElementType ElementType
	Content     []byte
}

// ParameterWithConfig contains the parameter and the configuration.
type ParameterWithConfig struct {
	Param  Parameter
	Config []byte // parameter configuration in Proto Buffer format
}

// Gradient is the gradient of the parameter.
type Gradient Parameter

// Service is the RPC service for pserver.
type Service struct {
	initialized chan struct{}

	mu       sync.Mutex
	opt      *optimizer
	paramMap map[string]Parameter

	etcdEndpoints string
	etcdClient    *clientv3.Client
	// etcdTimeout is also used as retry intervals.
	etcdTimeout time.Duration
	// desired number of pservers in the job.
	// assume desired will not change during one training job.
	desired int
	// FIXME: ensure GetExternalIP gets the correct ip for trainers to connect.
	externalIP string
}

// NewService creates a new service, will bypass etcd registration if no
// endpoints specified.
func NewService(endpoints string, numPservers int, timeout time.Duration) (*Service, error) {
	s := &Service{opt: newOptimizer(sgd, 0.005)}
	s.paramMap = make(map[string]Parameter)
	s.initialized = make(chan struct{})
	s.etcdEndpoints = endpoints
	s.etcdTimeout = timeout

	var err error
	s.externalIP, err = networkhelper.GetExternalIP()
	if err != nil {
		return nil, err
	}

	if endpoints != "" {
		// initialize connection to etcd, try
		ep := strings.Split(s.etcdEndpoints, ",")
		for {
			cli, err := clientv3.New(clientv3.Config{
				Endpoints:   ep,
				DialTimeout: s.etcdTimeout,
			})
			if err != nil {
				log.Errorf("connect to etcd error: %v", err)
				time.Sleep(s.etcdTimeout)
				continue
			}
			s.etcdClient = cli
			log.Debugf("inited client to %s", s.etcdEndpoints)
			break
		}
		// init /ps_desired using transaction, for multiple pservers may want to write
		// it at the same time.
		for {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			_, err := s.initDesiredPsercers(ctx, numPservers)
			cancel()
			if err != nil {
				log.Warn(err)
				time.Sleep(s.etcdTimeout)
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
			resp, err := s.etcdClient.Get(ctx, PsDesired)
			cancel()
			if err != nil {
				log.Errorf("getting %s error: %v", PsDesired, err)
				time.Sleep(s.etcdTimeout)
				continue
			}
			if len(resp.Kvs) != 0 {
				s.desired, err = strconv.Atoi(string(resp.Kvs[0].Value))
				if err != nil {
					log.Errorf("value of %s invalid %v\n", PsDesired, err)
					time.Sleep(s.etcdTimeout)
					// NOTE: wait util ps_desired value change
					continue
				}
				break
			}
		}
		// try register pserver node on etcd
		for {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			_, err := s.registerPserverEtcd(ctx)
			cancel()
			if err != nil {
				log.Warn(err)
				time.Sleep(s.etcdTimeout)
				continue
			}
			break
		}
	} // if endpoints != ""
	// Bypass etcd registration if no endpoints specified
	return s, nil
}

func (s *Service) initDesiredPsercers(ctx context.Context, numPservers int) (*clientv3.TxnResponse, error) {
	return concurrency.NewSTM(s.etcdClient, func(c concurrency.STM) error {
		dsStr := c.Get(PsDesired)
		if dsStr == "" {
			c.Put(PsDesired, strconv.Itoa(numPservers))
		}
		return nil
	}, concurrency.WithAbortContext(ctx), concurrency.WithIsolation(concurrency.RepeatableReads))
}

// registerPserverEtcd registers pserver node on etcd using transaction.
func (s *Service) registerPserverEtcd(ctx context.Context) (*clientv3.TxnResponse, error) {
	return concurrency.NewSTM(s.etcdClient, func(c concurrency.STM) error {
		registered := false
		for i := 0; i < s.desired; i++ {
			psKey := "/ps/" + strconv.Itoa(i)
			log.Debugf("checking %s", psKey)
			ps := c.Get(psKey)
			log.Debugf("got value (%s) for key: %s", ps, psKey)

			if ps == "" {
				resp, err := s.etcdClient.Grant(context.TODO(), 5)
				if err != nil {
					log.Fatal(err)
				}
				// find the first id and write info
				c.Put(psKey, s.externalIP, clientv3.WithLease(resp.ID))
				log.Debugf("set pserver node %s with value %s", psKey, s.externalIP)
				ch, kaerr := s.etcdClient.KeepAlive(context.TODO(), resp.ID)
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
				registered = true
				break
			}
		}
		if registered == true {
			return nil
		}
		return errors.New("not registerd, may due to already have enough pservers")
	}, concurrency.WithAbortContext(ctx), concurrency.WithIsolation(concurrency.RepeatableReads))
}

// InitParam initializes a parameter.
func (s *Service) InitParam(paramWithConfigs ParameterWithConfig, dummy *int) error {
	select {
	case <-s.initialized:
		return errors.New(AlreadyInitialized)
	default:
	}

	// TODO(helin): parse parameter config

	s.mu.Lock()
	defer s.mu.Unlock()

	// TODO(helin): check if paramWithConfigs.Param.Content is
	// properly memory aligned, if not, make copy to a memory
	// aligned region.
	s.paramMap[paramWithConfigs.Param.Name] = paramWithConfigs.Param
	return nil
}

// FinishInitParams tells the parameter server that the parameter
// initialization has finished.
func (s *Service) FinishInitParams(dummy0 int, dummy1 *int) error {
	select {
	case <-s.initialized:
		return errors.New(AlreadyInitialized)
	default:
	}

	close(s.initialized)
	return nil
}

// SendGrad sends gradient to parameter servers for parameter
// optimization.
func (s *Service) SendGrad(g Gradient, dummy *int) error {
	select {
	case <-s.initialized:
	default:
		return errors.New(Uninitialized)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	p, ok := s.paramMap[g.Name]
	if !ok {
		return fmt.Errorf("parameter: %s does not exist", g.Name)
	}

	return s.opt.UpdateParameter(p, g)
}

// GetParam gets parameters from the parameter server.
func (s *Service) GetParam(name string, parameter *Parameter) error {
	<-s.initialized
	s.mu.Lock()
	defer s.mu.Unlock()

	p, ok := s.paramMap[name]
	if !ok {
		return fmt.Errorf("parameter: %s does not exist", name)
	}

	// The parameter content (a byte slice) may change
	// during RPC serialization due to write from other
	// goroutine, we allow it since mini-batch based deep
	// learning optimization methods are stochastic in
	// nature. This race condition is allowed deliberately
	// to save the program from making a copy of the
	// paramter content.
	*parameter = p
	return nil
}

// Save tells the parameter server to save parameters.
func (s *Service) Save(path string, dummy *int) error {
	<-s.initialized

	// TODO
	return nil
}
