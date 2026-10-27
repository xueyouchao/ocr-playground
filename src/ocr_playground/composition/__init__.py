"""The two ends of the program: the dependency-injection container, and the adapters it wires.

`composition` is the only package allowed to know every other package. It is where configuration is
read, where a mode maps to an engine, and where an interface asks for a service it does not build
itself.
"""
