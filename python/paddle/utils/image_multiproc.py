import os, psutil
import cv2
from paddle.utils.image_util import *
import multiprocessing
import subprocess, signal, sys


class CvImageTransfomer(ImageTransformer):
    """
    CvImageTransfomer used python-opencv to process image.
    """

    def __init__(self,
                 min_size=None,
                 crop_size=None,
                 transpose=None,
                 channel_swap=None,
                 mean=None,
                 is_train=True,
                 is_color=True):
        ImageTransformer.__init__(self, transpose, channel_swap, mean, is_color)
        self.min_size = min_size
        self.crop_size = crop_size
        self.is_train = is_train

    def cv_resize_fixed_short_side(self, im, min_size):
        row, col = im.shape[:2]
        scale = min_size / float(min(row, col))
        if row < col:
            row = min_size
            col = int(round(col * scale))
            col = col if col > min_size else min_size
        else:
            col = min_size
            row = int(round(row * scale))
            row = row if row > min_size else min_size
        resized_size = row, col
        im = cv2.resize(im, resized_size, interpolation=cv2.INTER_CUBIC)
        return im

    def crop_img(self, im):
        """
        Return cropped image.
        The size of the cropped image is inner_size * inner_size.
        im: (H x W x K) ndarrays
        """
        row, col = im.shape[:2]
        start_h, start_w = 0, 0
        if self.is_train:
            start_h = np.random.randint(0, row - self.crop_size + 1)
            start_w = np.random.randint(0, col - self.crop_size + 1)
        else:
            start_h = (row - self.crop_size) / 2
            start_w = (col - self.crop_size) / 2
        end_h, end_w = start_h + self.crop_size, start_w + self.crop_size
        if self.is_color:
            im = im[start_h:end_h, start_w:end_w, :]
        else:
            im = im[start_h:end_h, start_w:end_w]
        if (self.is_train) and (np.random.randint(2) == 0):
            if self.is_color:
                im = im[:, ::-1, :]
            else:
                im = im[:, ::-1]
        return im

    def transform(self, im):
        im = self.cv_resize_fixed_short_side(im, self.min_size)
        im = self.crop_img(im)
        # transpose, swap channel, sub mean
        im = im.astype('float32')
        ImageTransformer.transformer(self, im)
        return im

    def load_image_from_string(self, data):
        flag = cv2.CV_LOAD_IMAGE_COLOR if self.is_color else cv2.CV_LOAD_IMAGE_GRAYSCALE
        im = cv2.imdecode(np.fromstring(data, np.uint8), flag)
        return im

    def transform_from_string(self, data):
        im = self.load_image_from_string(data)
        return self.transform(im)


class MultiProcessImageTransfomer():
    def __init__(self,
                 procnum=10,
                 capacity=10240,
                 min_size=None,
                 crop_size=None,
                 transpose=None,
                 channel_swap=None,
                 mean=None,
                 is_train=True,
                 is_color=True):
        self.procnum = procnum
        self.capacity = capacity
        self.size = 0
        self.count = 0
        signal.signal(signal.SIGTERM, self.kill_child_processes)
        self.fetch_queue = multiprocessing.Queue(maxsize=capacity)
        self.cv_transformer = CvImageTransfomer(min_size, crop_size, transpose,
                                                channel_swap, mean, is_train,
                                                is_color)

    def __del__(self):
        try:
            for p in self.procs:
                p.join()
        except Exception as e:
            print str(e)

    def reset(self, size):
        self.size = size
        self.count = 0
        self.procs = []

    def run_proc(self, data, label):
        dlen = len(label)
        self.reset(dlen)
        for i in xrange(self.procnum):
            start = dlen * i / self.procnum
            end = dlen * (i + 1) / self.procnum
            proc = multiprocessing.Process(
                target=self.batch_transfomer,
                args=(data[start:end], label[start:end]))
            proc.daemon = True
            self.procs.append(proc)
        for p in self.procs:
            p.start()

    def get(self):
        """
        Return one processed image.
        """
        # block if necessary until an item is available
        data, lab = self.fetch_queue.get(block=True)
        self.count += 1
        if self.count == self.size:
            try:
                for p in self.procs:
                    p.join()
            except Exception as e:
                print str(e)
        return data, lab

    def batch_transfomer(self, data, label):
        """
        param data: input data in format of image string
        type data: a list of string
        label: the label of image
        """
        for i in xrange(len(label)):
            res = self.cv_transformer.transform_from_string(data[i])
            self.fetch_queue.put((res, int(label[i])))

    def kill_child_processes(self, signum, frame):
        """
        Kill a process's child processes in python.
        """
        parent_id = os.getpid()
        ps_command = subprocess.Popen(
            "ps -o pid --ppid %d --noheaders" % parent_id,
            shell=True,
            stdout=subprocess.PIPE)
        ps_output = ps_command.stdout.read()
        retcode = ps_command.wait()
        for pid_str in ps_output.strip().split("\n")[:-1]:
            os.kill(int(pid_str), signal.SIGTERM)
        sys.exit()
