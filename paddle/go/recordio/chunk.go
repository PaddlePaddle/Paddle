package recordio

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"

	"github.com/golang/snappy"
)

// A Chunk contains the Header and optionally compressed records.  To
// create a chunk, just use ch := &Chunk{}.
type Chunk struct {
	records  [][]byte
	numBytes int // sum of record lengths.
}

func (ch *Chunk) add(record []byte) {
	ch.records = append(ch.records, record)
	ch.numBytes += len(record)
}

// dump the chunk into w, and clears the chunk and makes it ready for
// the next add invocation.
func (ch *Chunk) dump(w io.Writer, compressorIndex int) error {
	// NOTE: don't check ch.numBytes instead, because empty
	// records are allowed.
	if len(ch.records) == 0 {
		return nil
	}

	// Write raw records and their lengths into data buffer.
	var data bytes.Buffer

	for _, r := range ch.records {
		var rs [4]byte
		binary.LittleEndian.PutUint32(rs[:], uint32(len(r)))

		if _, e := data.Write(rs[:]); e != nil {
			return fmt.Errorf("Failed to write record length: %v", e)
		}

		if _, e := data.Write(r); e != nil {
			return fmt.Errorf("Failed to write record: %v", e)
		}
	}

	compressed, e := compressData(&data, compressorIndex)
	if e != nil {
		return e
	}

	// Write chunk header and compressed data.
	hdr := &Header{
		checkSum:       crc32.ChecksumIEEE(compressed.Bytes()),
		compressor:     uint32(compressorIndex),
		compressedSize: uint32(compressed.Len()),
		numRecords:     uint32(len(ch.records)),
	}

	if _, e := hdr.write(w); e != nil {
		return fmt.Errorf("Failed to write chunk header: %v", e)
	}

	if _, e := w.Write(compressed.Bytes()); e != nil {
		return fmt.Errorf("Failed to write chunk data: %v", e)
	}

	// Clear the current chunk.
	ch.records = nil
	ch.numBytes = 0

	return nil
}

type noopCompressor struct {
	*bytes.Buffer
}

func (c *noopCompressor) Close() error {
	return nil
}

func compressData(src io.Reader, compressorIndex int) (*bytes.Buffer, error) {
	compressed := new(bytes.Buffer)
	var compressor io.WriteCloser

	switch compressorIndex {
	case NoCompression:
		compressor = &noopCompressor{compressed}
	case Snappy:
		compressor = snappy.NewBufferedWriter(compressed)
	case Gzip:
		compressor = gzip.NewWriter(compressed)
	default:
		return nil, fmt.Errorf("Unknown compression algorithm: %d", compressorIndex)
	}

	if _, e := io.Copy(compressor, src); e != nil {
		return nil, fmt.Errorf("Failed to compress chunk data: %v", e)
	}
	compressor.Close()

	return compressed, nil
}

// parse the specified chunk from r.
func parseChunk(r io.ReadSeeker, chunkOffset int64) (*Chunk, error) {
	var e error
	var hdr *Header

	if _, e = r.Seek(chunkOffset, io.SeekStart); e != nil {
		return nil, fmt.Errorf("Failed to seek chunk: %v", e)
	}

	hdr, e = parseHeader(r)
	if e != nil {
		return nil, fmt.Errorf("Failed to parse chunk header: %v", e)
	}

	var buf bytes.Buffer
	if _, e = io.CopyN(&buf, r, int64(hdr.compressedSize)); e != nil {
		return nil, fmt.Errorf("Failed to read chunk data: %v", e)
	}

	if hdr.checkSum != crc32.ChecksumIEEE(buf.Bytes()) {
		return nil, fmt.Errorf("Checksum checking failed.")
	}

	deflated, e := deflateData(&buf, int(hdr.compressor))
	if e != nil {
		return nil, e
	}

	ch := &Chunk{}
	for i := 0; i < int(hdr.numRecords); i++ {
		var rs [4]byte
		if _, e = deflated.Read(rs[:]); e != nil {
			return nil, fmt.Errorf("Failed to read record length: %v", e)
		}

		r := make([]byte, binary.LittleEndian.Uint32(rs[:]))
		if _, e = deflated.Read(r); e != nil {
			return nil, fmt.Errorf("Failed to read a record: %v", e)
		}

		ch.records = append(ch.records, r)
		ch.numBytes += len(r)
	}

	return ch, nil
}

func deflateData(src io.Reader, compressorIndex int) (*bytes.Buffer, error) {
	var e error
	var deflator io.Reader

	switch compressorIndex {
	case NoCompression:
		deflator = src
	case Snappy:
		deflator = snappy.NewReader(src)
	case Gzip:
		deflator, e = gzip.NewReader(src)
		if e != nil {
			return nil, fmt.Errorf("Failed to create gzip reader: %v", e)
		}
	default:
		return nil, fmt.Errorf("Unknown compression algorithm: %d", compressorIndex)
	}

	deflated := new(bytes.Buffer)
	if _, e = io.Copy(deflated, deflator); e != nil {
		return nil, fmt.Errorf("Failed to deflate chunk data: %v", e)
	}

	return deflated, nil
}
