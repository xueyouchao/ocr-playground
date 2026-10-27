"""Failures that belong to a running engine rather than to the request.

The GLM endpoint being unreachable is not the uploader's mistake, so it is not a `DomainError`. The
HTTP adapter is the only layer that turns these into status codes; nothing below it imports
`http.HTTPStatus`.
"""
from __future__ import annotations


class ApplicationError(Exception):
    """The reading could not be performed by an engine that was expected to work."""


class EngineUnavailable(ApplicationError):
    """The engine is not configured on this deployment — a 503, with the alternative named."""


class EngineTimeout(ApplicationError):
    """The engine did not answer inside the document's budget — a 504."""


class EngineFailure(ApplicationError):
    """The engine answered, but not with a reading — a 502."""
