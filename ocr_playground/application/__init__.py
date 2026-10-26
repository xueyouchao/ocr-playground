"""Use cases, expressed in domain terms.

An application service orchestrates ports: render this document, read each page with this engine,
attach a preview if asked. It does not know whether the renderer is poppler or a fake, whether the
engine is a subprocess or a GPU endpoint, or that an evidence JSON even exists.
"""
