"""HTTP: the only place that knows what a status code is.

Adapters raise domain or application errors; this package maps them to responses and owns the
transport-level rules (CORS, caching, the body size cap). Nothing below it imports `HTTPStatus`.
"""
