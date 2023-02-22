from threading import Thread, Event
from queue import Queue
import cv2 as cv

class ThreadWriter(Thread):
    def __init__(self):
        self.queue = Queue()
        self.event = Event()
        Thread.__init__(self)

    def write_file(self, path: str, data: bytes):
        self.queue.put((path, data))
        return self.queue.qsize()

    def close(self, wait=True):
        if wait:
            while not self.queue.empty():
                continue
        self.event.set()

    def run(self):
        while not self.event.is_set():
            if self.queue.empty():
                continue
            path, data = self.queue.get()
            """
            with open(path, 'wb') as fp:
                fp.write(data)
            """
            cv.imwrite(path, data)
            #print("data written")