"""Utilities for building contextual functions.

This module provides decorators for functions so that they can use contextual
arguments and a mixin for classes that provides a `get_new_context` method
which could be mapped to `__call__` to produce and use concepts as in the
previous example.

This module was originally developed as part of Rig (https://github.com/project-rig/rig)
and was developed by Andrew Mundy (andrew.mundy@ieee.org)
"""
import collections
import inspect
import functools
import sentinel
from six import iteritems

Required = sentinel.create('Required')
"""Allow specifying keyword arguments as required, i.e., they must be satisfied
by either the context OR by the caller.

This is useful when a method has optional parameters and contextual arguments::

    @ContextMixin.use_contextual_arguments()
    def sdram_alloc(self, size, tag=0, x=Required, y=Required):
        # ...

And also when using non-default keyword-only are required::

    @ContextMixin.use_contextual_arguments(app_id=Required)
    def example(*args, **kwargs):
        app_id = kwargs.pop(app_id)  # Must always be given
        # ...
"""


class ContextMixin(object):
    """A mix-in which provides a context stack and allows querying of the stack
    to form keyword arguments.
    """
    def __init__(self, initial_context={}):
        """Create a context stack for this object.

        Parameters
        ----------
        initial_context : {kwarg: value}
            An initial set of contextual arguments mapping keyword to value.
        """
        self.__context_stack = collections.deque()
        self.__context_stack.append(Context(initial_context))

    def get_new_context(self, **kwargs):
        """Create a new context with the given keyword arguments."""
        return Context(kwargs, self.__context_stack)

    def update_current_context(self, **context_args):
        """Update the current context to contain new arguments."""
        self.__context_stack[-1].update(context_args)

    def get_context_arguments(self):
        """Return a dictionary containing the current context arguments."""
        cargs = {}
        for context in self.__context_stack:
            cargs.update(context.context_arguments)
        return cargs

    @staticmethod
    def use_contextual_arguments(**kw_only_args_defaults):
        """Decorator function which allows the wrapped function to accept
        arguments not specified in the call from the context.

        Arguments whose default value is set to the Required sentinel must be
        supplied either by the context or the caller and a TypeError is raised
        if not.

        .. warning::
            Due to a limitation in the Python 2 version of the introspection
            library, this decorator only works with functions which do not have
            any keyword-only arguments. For example this function cannot be
            handled::

                def f(*args, kw_only_arg=123)

            Note, however, that the decorated function *can* accept and pass-on
            keyword-only arguments specified via `**kw_only_args_defaults`.

        Parameters
        ----------
        **kw_only_args_defaults : {name: default, ...}
            Specifies the set of keyword-only arguments (and their default
            values) accepted by the underlying function. These will be passed
            via the kwargs to the underlying function, e.g.::

                @ContextMixin.use_contextual_arguments(kw_only_arg=123)
                def f(self, **kwargs):
                    kw_only_arg = kwargs.pop("kw_only_arg")

                # Wrapped function can be called with keyword-only-arguments:
                spam.f(*[], kw_only_arg=12)

            Keyword-only arguments can be made mandatory by setting their
            default value to the Required sentinel.
        """
        def decorator(f):
            # Extract any positional and positional-and-key-word arguments
            # which may be set.
            arg_names, varargs, keywords, defaults = inspect.getargspec(f)

            # Sanity check: non-keyword-only arguments should't be present in
            # the keyword-only-arguments list.
            assert set(keywords or {}).isdisjoint(set(kw_only_args_defaults))

            # Fully populate the default argument values list, setting the
            # default for mandatory arguments to the 'Required' sentinel.
            if defaults is None:
                defaults = []
            defaults = (([Required] * (len(arg_names) - len(defaults))) +
                        list(defaults))

            # Update the docstring signature to include the specified arguments
            #@add_signature_to_docstring(f, kw_only_args=kw_only_args_defaults)
            @functools.wraps(f)
            def f_(self, *args, **kwargs):
                # Construct a dictionary of arguments (and their default
                # values) which may potentially be set by the context. This
                # includes any non-supplied positional arguments and any
                # keyword-only arguments.
                new_kwargs = dict(zip(arg_names[1 + len(args):],
                                      defaults[1 + len(args):]))
                new_kwargs.update(kw_only_args_defaults)

                # Values from the context take priority over default argument
                # values.
                context = self.get_context_arguments()
                for name, val in iteritems(context):
                    if name in new_kwargs:
                        new_kwargs[name] = val

                # Finally, the values actually pased to the function call take
                # ultimate priority.
                new_kwargs.update(kwargs)

                # Raise a TypeError if any `Required` sentinels remain
                for k, v in iteritems(new_kwargs):
                    if v is Required:
                        raise TypeError(
                            "{!s}: missing argument {}".format(f.__name__, k))

                return f(self, *args, **new_kwargs)
            return f_

        return decorator


class Context(object):
    """A context object that stores arguments that may be passed to
    functions.
    """
    def __init__(self, context_arguments, stack=None):
        """Create a new context object that can be added to a stack.

        Parameters
        ----------
        context_arguments : {kwarg: value}
            A dict of contextual arguments mapping keyword to value.
        stack : :py:class:`deque`
            Context stack to which this context will append itself when
            entered.
        """
        self.context_arguments = dict(context_arguments)
        self.stack = stack
        self._before_close = list()

    def update(self, updates):
        """Update the arguments contained within this context."""
        self.context_arguments.update(updates)

    def before_close(self, *args):
        """Call the given function(s) before this context is exited."""
        for fn in args:
            self._before_close.append(fn)

    def __enter__(self):
        # Add this context object to the stack
        self.stack.append(self)

    def __exit__(self, exception_type, exception_value, traceback):
        try:
            # Call all the passed functions before closing the context
            for fn in self._before_close:
                fn()
        finally:
            # Remove self from the stack
            assert self.stack.pop() is self
