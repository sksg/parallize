import numpy as np
from numpy.core.numerictypes import typecodes
import inspect
import functools
import re
import builtins
import os
from concurrent.futures import ThreadPoolExecutor as thread_pool
from concurrent.futures import ProcessPoolExecutor as process_pool
from concurrent.futures import as_completed


def _iterable(y):
    try:
        iter(y)
    except TypeError:
        return False
    return True

# We use an extended version of:
# http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
_DIMENSION_NAME = r'\w+'
_CORE_DIMENSION_LIST = '(?:{0:}(?:,{0:})*)?'.format(_DIMENSION_NAME)
_VECTOR_ARGUMENT = r'(\({}\))'.format(_CORE_DIMENSION_LIST)
_EXCLUDED_ARGUMENT = r'(_)'
_ARGUMENT = r'(?:{0:}|{1:})'.format(_VECTOR_ARGUMENT, _EXCLUDED_ARGUMENT)
_ARGUMENT_LIST = '{0:}(?:,{0:})*'.format(_ARGUMENT)
_OUT_ARGUMENT_LIST = '{0:}(?:,{0:})*'.format(_VECTOR_ARGUMENT)
_SIGNATURE = '^{0:}->{1:}$'.format(_ARGUMENT_LIST, _OUT_ARGUMENT_LIST)


def _parse_signature(signature):
    if not re.match(_SIGNATURE, signature):
        raise ValueError(
            'not a valid gufunc signature: {}'.format(signature))
    inargs, outargs = [], []
    _in, _out = signature.split('->')
    for arg in re.findall(_ARGUMENT, _in):
        if arg[1] == "_":
            inargs.append(None)
        else:
            inarg = []
            for match in re.findall(_DIMENSION_NAME, arg[0]):
                try:
                    inarg.append(int(match))
                except:
                    inarg.append(match)
            inargs.append(tuple(inarg))

    for arg in re.findall(_ARGUMENT, _out):
        if arg[1] == "_":
            outargs.append(None)
        else:
            outarg = []
            for match in re.findall(_DIMENSION_NAME, arg[0]):
                try:
                    outarg.append(int(match))
                except:
                    outarg.append(match)
            outargs.append(tuple(outarg))
    return inargs, outargs


def _update_dim_sizes(dim_sizes, arg, core_dims):
    if not core_dims:
        return
    num_core_dims = len(core_dims)
    if arg.ndim < num_core_dims:
        raise ValueError('%d-dimensional argument does not have enough '
                         'dimensions for all core dimensions %r'
                         % (arg.ndim, core_dims))
    core_shape = arg.shape[-num_core_dims:]
    for dim, size in zip(core_dims, core_shape):
        if dim in dim_sizes:
            if size != dim_sizes[dim]:
                raise ValueError('inconsistent size for core dimension'
                                 ' %r: %r vs %r'
                                 % (dim, size, dim_sizes[dim]))
        elif isinstance(dim, str):
            dim_sizes[dim] = size
        elif dim != size:
            raise ValueError('inconsistent size for core dimension: %r vs %r'
                             % (dim, size))


def _parse_input_dimensions(args, arg_dims):
    dim_sizes = {}
    broadcast_args = []
    for a, dims in zip(args, arg_dims):
        if dims is None:
            broadcast_args.append(None)
            continue
        _update_dim_sizes(dim_sizes, a, dims)
        ndim = a.ndim - len(dims)
        dummy_array = np.lib.stride_tricks.as_strided(0, a.shape[:ndim])
        broadcast_args.append(dummy_array)
    broadcast_shape = np.lib.stride_tricks._broadcast_shape(*broadcast_args)
    return broadcast_shape, dim_sizes


def _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims):
    return [(broadcast_shape + tuple((dim_sizes[dim]
                                      if isinstance(dim, str) else dim)
                                     for dim in core_dims)
             if core_dims is not None else None)
            for core_dims in list_of_core_dims]


def _create_arrays(broadcast_shape, dim_sizes, list_of_core_dims, dtypes):
    shapes = _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims)
    arrays = tuple(np.empty(shape, dtype=dtype)
                   for shape, dtype in zip(shapes, dtypes))
    return arrays


def parallize(signature, otypes=None, doc=None, default='parallelenv',
              evn='MEGA_PARALLIZE', isvec=False, parallel='threads',
              sendindex=False):
    def wrap_parallized(pyfunc):
        return parallized(pyfunc, signature, otypes, doc, default,
                          evn, isvec, parallel, sendindex)
    return wrap_parallized


class parallized(object):  # inspired by np.vectorize
    def __init__(self, pyfunc, signature, otypes=None, doc=None,
                 default='parallel', evn='MEGA_PARALLIZE', isvec=False,
                 parallel_type='threads', sendindex=False):
        self.signature = signature
        self.default = default
        self.evn = evn
        self.isvec = isvec
        self.parallel_type = parallel_type
        self.sendindex = sendindex
        self._ufunc = None    # Caching to improve default performance

        if doc is not None:
            self.__doc__ = doc
        else:
            self.__doc__ = pyfunc.__doc__

        if isinstance(otypes, str):
            for char in otypes:
                if char not in typecodes['All']:
                    raise ValueError("Invalid otype specified: %s" % (char,))
        elif _iterable(otypes):
            otypes = ''.join([np.dtype(x).char for x in otypes])
        elif otypes is not None:
            raise ValueError("Invalid otype specification")
        self.otypes = otypes

        self._in, self._out = _parse_signature(signature)
        self.excluded = [(a is None) for a in self._in]

        self.pyfunc = pyfunc
        self.__wrapped__ = pyfunc
        self.parameters = [k for k in inspect.signature(pyfunc).parameters]
        if self.sendindex:
            self.parameters = self.parameters[1:]

    def _process_args(self, args, kwargs):
        givenargs = list(args)
        allargs = []
        for p in self.parameters:
            if p in kwargs:
                allargs.append(kwargs.pop(p))
            else:
                if len(args) == 0:
                    msg = 'expected {}, got {}'.format(len(self.parameters),
                                                       len(givenargs))
                    raise TypeError("Missing positional arguments: " + msg)
                allargs.append(args[0])
                args = args[1:]

        if len(kwargs) != 0:
            raise TypeError("Unknown keyword arguments {}!".format(kwargs))
        if len(args) != 0:
            msg = 'expected {}, got {}'.format(len(self.parameters),
                                               len(givenargs))
            raise TypeError("Too many positional arguments: " + msg)

        args = tuple((np.asanyarray(a) if not ex else a)
                     for a, ex in zip(allargs, self.excluded))

        broadcast_shape, dim_sizes = _parse_input_dimensions(args, self._in)
        input_shapes = _calculate_shapes(broadcast_shape, dim_sizes, self._in)
        args = [(np.broadcast_to(arg, shape, subok=True)
                 if shape is not None else arg)
                for arg, shape in zip(args, input_shapes)]
        return broadcast_shape, dim_sizes, args

    def __call__(self, *args, **kwargs):
        if self.default is 'parallel':
            return self.parallel(*args, **kwargs)
        if self.default is 'sequential':
            return self.sequential(*args, **kwargs)
        if self.default is 'vectorized':
            return self.vectorized(*args, **kwargs)
        if self.default is 'parallelenv':
            if self.evn in os.environ and not os.environ[self.evn]:
                return self.vectorized(*args, **kwargs)
            else:
                return self.parallel(*args, **kwargs)

    def vectorized(self, *args, **kwargs):
        if self.isvec:
            if self.sendindex:
                return self.pyfunc(None, *args, **kwargs)
            else:
                return self.pyfunc(*args, **kwargs)
        else:
            return self.sequential(*args, **kwargs)

    def sequential(self, *args, **kwargs):
        broadcast_shape, dim_sizes, args = self._process_args(args, kwargs)

        outputs = None
        otypes = self.otypes
        nout = len(self._out)

        for index in np.ndindex(*broadcast_shape):
            i_args = ((arg[index] if _in is not None else arg)
                      for _in, arg in zip(self._in, args))
            if self.sendindex:
                results = self.pyfunc(index, *i_args)
            else:
                results = self.pyfunc(*i_args)

            n_results = len(results) if isinstance(results, tuple) else 1

            if nout != n_results:
                raise ValueError(
                    'wrong number of outputs from pyfunc: expected %r, got %r'
                    % (nout, n_results))

            if nout == 1:
                results = (results,)

            if outputs is None:
                for result, core_dims in zip(results, self._out):
                    _update_dim_sizes(dim_sizes, result, core_dims)

                if otypes is None:
                    otypes = [np.asarray(result).dtype for result in results]

                outputs = _create_arrays(broadcast_shape, dim_sizes,
                                         self._out, otypes)

            for output, result in zip(outputs, results):
                output[index] = result

        if outputs is None:
            # did not call the function even once
            if otypes is None:
                raise ValueError('cannot call `vectorize` on size 0 inputs '
                                 'unless `otypes` is set')
            if builtins.any(dim not in dim_sizes
                            for dims in self._out
                            for dim in dims):
                raise ValueError('cannot call `vectorize` with a signature '
                                 'including new output dimensions on size 0 '
                                 'inputs')
            outputs = _create_arrays(broadcast_shape, dim_sizes,
                                     self._out, otypes)

        return outputs[0] if nout == 1 else outputs

    def parallel(self, *args, **kwargs):
        broadcast_shape, dim_sizes, args = self._process_args(args, kwargs)

        outputs = None
        otypes = self.otypes
        nout = len(self._out)

        if self.parallel_type == 'threads':
            pool = thread_pool(os.cpu_count())
        elif self.parallel_type == 'processes':
            pool = process_pool(os.cpu_count())
        futures = {}

        for index in np.ndindex(*broadcast_shape):
            i_args = ((arg[index] if _in is not None else arg)
                      for _in, arg in zip(self._in, args))
            if self.sendindex:
                futures[pool.submit(self.pyfunc, index, *i_args)] = index
            else:
                futures[pool.submit(self.pyfunc, *i_args)] = index

        for f in as_completed(futures):
            index = futures[f]
            results = f.result()

            n_results = len(results) if isinstance(results, tuple) else 1

            if nout != n_results:
                raise ValueError(
                    'wrong number of outputs from pyfunc: expected %r, got %r'
                    % (nout, n_results))

            if nout == 1:
                results = (results,)

            if outputs is None:
                for result, core_dims in zip(results, self._out):
                    _update_dim_sizes(dim_sizes, result, core_dims)

                if otypes is None:
                    otypes = [np.asarray(result).dtype for result in results]

                outputs = _create_arrays(broadcast_shape, dim_sizes,
                                         self._out, otypes)

            for output, result in zip(outputs, results):
                output[index] = result

        if outputs is None:
            # did not call the function even once
            if otypes is None:
                raise ValueError('cannot call `vectorize` on size 0 inputs '
                                 'unless `otypes` is set')
            if builtins.any(dim not in dim_sizes
                            for dims in self._out
                            for dim in dims):
                raise ValueError('cannot call `vectorize` with a signature '
                                 'including new output dimensions on size 0 '
                                 'inputs')
            outputs = _create_arrays(broadcast_shape, dim_sizes,
                                     self._out, otypes)

        return outputs[0] if nout == 1 else outputs


class asparallel(object):
    def __init__(self, pyfunc, default='parallelenv', evn='MEGA_PARALLIZE'):
        self.pyfunc = pyfunc
        self.default = default
        self.evn = evn
        self.__wrapped__ = pyfunc

    def __call__(self, *args, **kwargs):
        if self.default is 'parallel':
            return self.parallel(*args, **kwargs)
        if self.default is 'sequential':
            return self.sequential(*args, **kwargs)
        if self.default is 'vectorized':
            return self.vectorized(*args, **kwargs)
        if self.default is 'parallelenv':
            if self.evn in os.environ and not os.environ[self.evn]:
                return self.vectorized(*args, **kwargs)
            else:
                return self.parallel(*args, **kwargs)

    def parallel(self, *args, **kwargs):
        def wrap_parallels(parallelfunc):
            return parallelfunc.parallel
        return self.pyfunc(wrap_parallels, *args, **kwargs)

    def sequential(self, *args, **kwargs):
        def wrap_parallels(parallelfunc):
            return parallelfunc.sequential
        return self.pyfunc(wrap_parallels, *args, **kwargs)

    def vectorized(self, *args, **kwargs):
        def wrap_parallels(parallelfunc):
            return parallelfunc.vectorized
        return self.pyfunc(wrap_parallels, *args, **kwargs)
