# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/landmark2face_wy/sync_batchnorm/comm.py
# Compiled at: 2024-03-06 17:51:32
# Size of source mod 2**32: 4449 bytes
import queue, collections, threading
__all__ = [
 "FutureResult", "SlavePipe", "SyncMaster"]

class FutureResult(object):
    __doc__ = "A thread-safe future implementation. Used only as one-to-one pipe."

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, "Previous result has't been fetched."
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()
            res = self._result
            self._result = None
            return res


_MasterRegistry = collections.namedtuple("MasterRegistry", ["result"])
_SlavePipeBase = collections.namedtuple("_SlavePipeBase", ["identifier", "queue", "result"])

class SlavePipe(_SlavePipeBase):
    __doc__ = "Pipe for master-slave communication."

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


class SyncMaster(object):
    __doc__ = "An abstract `SyncMaster` object.\n\n    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should\n    call `register(id)` and obtain an `SlavePipe` to communicate with the master.\n    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,\n    and passed to a registered callback.\n    - After receiving the messages, the master device should gather the information and determine to message passed\n    back to each slave devices.\n    "

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {"master_callback": (self._master_callback)}

    def __setstate__(self, state):
        self.__init__(state["master_callback"])

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(), "Queue is not clean before next initialization."
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True
        intermediates = [
         (
          0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())

        results = self._master_callback(intermediates)
        assert results[0][0] == 0, "The first result should belongs to the master."
        for (i, res) in results:
            if i == 0:
                pass
            else:
                self._registry[i].result.put(res)

        for i in range(self.nr_slaves):
            if not self._queue.get() is True:
                raise AssertionError
            return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)

