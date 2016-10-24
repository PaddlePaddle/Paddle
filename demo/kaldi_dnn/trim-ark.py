#!/usr/bin/env python
#from paddle.trainer_config_helpers import *
import sys
import struct

def trim_ark(file_in, file_out, num_utt):
    const_tag = struct.pack('6c', *[chr(i) for i in (0x00, 0x42, 0x46, 0x4d, 0x20, 0x04)])
    with open(file_in) as fin:
        fout = open(file_out, 'w')
        key = ''
        if num_utt > 0:
            while num_utt > 0:
                num_utt -= 1
                key = ''
                while True:
                    buf = fin.read(1)
                    if buf == ' ':
                        break
                    key = key + buf
                print key
                key += ' '
                fout.write(key)
                buf = fin.read(15)
                assert buf[:6] == const_tag and buf[10] == chr(0x04)
                fout.write(buf)
                num_frame = struct.unpack('<i', buf[6:10])[0]
                dim = struct.unpack('<i', buf[11:15])[0]
                buf = fin.read(4 * dim * num_frame) # 4 is size of float
                fout.write(buf)
        else:
            while num_utt < 0:
                num_utt += 1
                key = ''
                while True:
                    buf = fin.read(1)
                    if buf == ' ':
                        break
                    key = key + buf
                key += ' '
                buf = fin.read(15)
                assert buf[:6] == const_tag and buf[10] == chr(0x04)
                num_frame = struct.unpack('<i', buf[6:10])[0]
                dim = struct.unpack('<i', buf[11:15])[0]
                buf = fin.read(4 * dim * num_frame) # 4 is size of float
            while True:
                key = ''
                while True:
                    buf = fin.read(1)
                    if buf == '':
                        return
                    if buf == ' ':
                        break
                    key = key + buf
                print key
                key += ' '
                fout.write(key)
                buf = fin.read(15)
                assert buf[:6] == const_tag and buf[10] == chr(0x04)
                fout.write(buf)
                num_frame = struct.unpack('<i', buf[6:10])[0]
                dim = struct.unpack('<i', buf[11:15])[0]
                buf = fin.read(4 * dim * num_frame) # 4 is size of float
                fout.write(buf)


def main():
    if len(sys.argv) != 4:
        print >> sys.stderr, 'Usage: %s input.ark output.ark num_utt' % sys.argv[0]
        sys.exit(-1)
    trim_ark(sys.argv[1], sys.argv[2], int(sys.argv[3]))

if __name__ == '__main__':
    main()
