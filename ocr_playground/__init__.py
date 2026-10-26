"""The reading, with no HTTP, no subprocess and no GPU around it.

`domain` holds the vocabulary the rest of the program is allowed to use: what a document is, what one
page of reading looks like, what an engine is, and which failures are *the request's* fault. It knows
nothing about JSON, sockets, CocoIndex or poppler; those live in `infrastructure` and `interfaces`.

The dependency rule points inward: `domain` imports nothing from this package except itself.
"""
