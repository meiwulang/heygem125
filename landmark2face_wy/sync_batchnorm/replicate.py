# decompyle3 version 3.9.2
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.19 (default, Mar 20 2024, 15:27:52) 
# [Clang 14.0.6 ]
# Embedded file name: /code/landmark2face_wy/sync_batchnorm/replicate.py
# Compiled at: 2024-03-06 17:51:32
# Size of source mod 2**32: 3226 bytes
import functools
from torch.nn.parallel.data_parallel import DataParallel
__all__ = [
 "CallbackContext",
 "execute_replication_callbacks",
 "DataParallelWithCallback",
 "patch_replication_callback"]

class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, "__data_parallel_replicate__"):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    __doc__ = "\n    Data Parallel with a replication callback.\n\n    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by\n    original `replicate` function.\n    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`\n\n    Examples:\n        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)\n        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])\n        # sync_bn.__data_parallel_replicate__ will be invoked.\n    "

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


def patch_replication_callback(data_parallel):
    """
    Monkey-patch an existing `DataParallel` object. Add the replication callback.
    Useful when you have customized `DataParallel` implementation.

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
        > patch_replication_callback(sync_bn)
        # this is equivalent to
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
    """
    assert isinstance(data_parallel, DataParallel)
    old_replicate = data_parallel.replicate

    @functools.wraps(old_replicate)
    def new_replicate(module, device_ids):
        modules = old_replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules

    data_parallel.replicate = new_replicate

