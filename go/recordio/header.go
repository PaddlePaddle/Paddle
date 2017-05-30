package recordio

import (
	"encoding/binary"
	"fmt"
	"io"
)

const (
	// NoCompression means writing raw chunk data into files.
	// With other choices, chunks are compressed before written.
	NoCompression = iota
	// Snappy had been the default compressing algorithm widely
	// used in Google.  It compromises between speech and
	// compression ratio.
	Snappy
	// Gzip is a well-known compression algorithm.  It is
	// recommmended only you are looking for compression ratio.
	Gzip

	magicNumber       uint32 = 0x01020304
	defaultCompressor        = Snappy
)

// Header is the metadata of Chunk.
type Header struct {
	checkSum       uint32
	compressor     uint32
	compressedSize uint32
	numRecords     uint32
}

func (c *Header) write(w io.Writer) (int, error) {
	var buf [20]byte
	binary.LittleEndian.PutUint32(buf[0:4], magicNumber)
	binary.LittleEndian.PutUint32(buf[4:8], c.checkSum)
	binary.LittleEndian.PutUint32(buf[8:12], c.compressor)
	binary.LittleEndian.PutUint32(buf[12:16], c.compressedSize)
	binary.LittleEndian.PutUint32(buf[16:20], c.numRecords)
	return w.Write(buf[:])
}

func parseHeader(r io.Reader) (*Header, error) {
	var buf [20]byte
	if _, e := r.Read(buf[:]); e != nil {
		return nil, e
	}

	if v := binary.LittleEndian.Uint32(buf[0:4]); v != magicNumber {
		return nil, fmt.Errorf("Failed to parse magic number")
	}

	return &Header{
		checkSum:       binary.LittleEndian.Uint32(buf[4:8]),
		compressor:     binary.LittleEndian.Uint32(buf[8:12]),
		compressedSize: binary.LittleEndian.Uint32(buf[12:16]),
		numRecords:     binary.LittleEndian.Uint32(buf[16:20]),
	}, nil
}
