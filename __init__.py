from __future__ import absolute_import
from .lib import *
try:
    import copy_reg
except ImportError:
    import copyreg as copy_reg
from types import *
from sys import version_info
py3 = True if version_info > (3,) else False


def _pickle_method(method):
    func_name = method.__func__.__name__ if py3 else method.im_func.__name__
    obj = method.__self__ if py3 else method.im_self
    cls = method.__self__.__class__ if py3 else method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'):
        cls_name = cls.__name__.lstrip('_')
        if cls_name: func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(MethodType, _pickle_method, _unpickle_method)
