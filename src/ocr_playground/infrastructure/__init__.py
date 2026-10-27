"""Adapters: the only code that knows about poppler, ImageMagick, subprocesses and sockets.

Each module here implements a port from `domain.ports` and is selected by `composition.container`.
Importing an adapter must not start work: no endpoint is contacted, no tool is probed and no thread is
started until a use case asks for something.
"""
