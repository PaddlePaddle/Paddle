package recordio

import (
	"fmt"
	"os"
	"path/filepath"
)

// MultiScanner is a scanner for multiple recordio files.
type MultiScanner struct {
	paths      []string
	curFile    *os.File
	curScanner *Scanner
	pathIdx    int
	end        bool
	err        error
}

// NewMultiScanner creates a new MultiScanner.
func NewMultiScanner(paths []string) (*MultiScanner, error) {
	var ps []string
	for _, s := range paths {
		match, err := filepath.Glob(s)
		if err != nil {
			return nil, err
		}

		ps = append(ps, match...)
	}

	if len(ps) == 0 {
		return nil, fmt.Errorf("no valid path provided: %v", paths)
	}

	return &MultiScanner{paths: ps}, nil
}

// Scan moves the cursor forward for one record and loads the chunk
// containing the record if not yet.
func (s *MultiScanner) Scan() bool {
	if s.err != nil {
		return false
	}

	if s.end {
		return false
	}

	if s.curScanner == nil {
		more, err := s.nextFile()
		if err != nil {
			s.err = err
			return false
		}

		if !more {
			s.end = true
			return false
		}
	}

	curMore := s.curScanner.Scan()
	s.err = s.curScanner.Err()

	if s.err != nil {
		return curMore
	}

	if !curMore {
		err := s.curFile.Close()
		if err != nil {
			s.err = err
			return false
		}
		s.curFile = nil

		more, err := s.nextFile()
		if err != nil {
			s.err = err
			return false
		}

		if !more {
			s.end = true
			return false
		}

		return s.Scan()
	}
	return true
}

// Err returns the first non-EOF error that was encountered by the
// Scanner.
func (s *MultiScanner) Err() error {
	return s.err
}

// Record returns the record under the current cursor.
func (s *MultiScanner) Record() []byte {
	if s.curScanner == nil {
		return nil
	}

	return s.curScanner.Record()
}

// Close release the resources.
func (s *MultiScanner) Close() error {
	s.curScanner = nil
	if s.curFile != nil {
		err := s.curFile.Close()
		s.curFile = nil
		return err
	}
	return nil
}

func (s *MultiScanner) nextFile() (bool, error) {
	if s.pathIdx >= len(s.paths) {
		return false, nil
	}

	path := s.paths[s.pathIdx]
	s.pathIdx++
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}

	idx, err := LoadIndex(f)
	if err != nil {
		f.Close()
		return false, err
	}

	s.curFile = f
	s.curScanner = NewScanner(f, idx, 0, -1)
	return true, nil
}
