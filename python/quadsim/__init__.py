"""quadsim: the Rust simulator, as a Python package.

The compiled code lives in the extension module `quadsim._quadsim` (built from
crates/quadsim by maturin). This file re-exports it so callers write
`import quadsim` and never need to know about the underscore module.
"""

from ._quadsim import add_arrays

__all__ = ["add_arrays"]
