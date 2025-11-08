from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import pybind11


class BuildExt(build_ext):
    """Customize compiler options for different toolchains."""

    c_opts = {
        "msvc": ["/EHsc", "/std:c++17"],
        "unix": ["-std=c++17", "-O3"],
    }

    l_opts = {
        "msvc": [],
        "unix": [],
    }

    if sys.platform == "darwin":
        c_opts["unix"].append("-mmacosx-version-min=10.15")
        l_opts["unix"].append("-mmacosx-version-min=10.15")

    def build_extensions(self) -> None:
        ct = self.compiler.compiler_type
        for ext in self.extensions:
            ext.extra_compile_args = ext.extra_compile_args or []
            ext.extra_link_args = ext.extra_link_args or []
            ext.extra_compile_args.extend(self.c_opts.get(ct, []))
            ext.extra_link_args.extend(self.l_opts.get(ct, []))
            if ct != "msvc" and sys.platform != "win32":
                ext.extra_link_args.append("-lpthread")
        super().build_extensions()


here = Path(__file__).parent.resolve()

core_sources = [
    "src/gng/gng_module.cpp",
    "src/gng/gng_server.cpp",
    "src/gng/gng_algorithm.cpp",
    "src/utils/threading.cpp",
    "src/utils/tinythread.cpp",
]

include_dirs = [
    str(here / "inst/include"),
    str(here / "inst/include/gng"),
    str(here / "inst/include/utils"),
    pybind11.get_include(False),
    pybind11.get_include(True),
]


ext_modules = [
    Extension(
        "gng._core",
        sources=core_sources,
        include_dirs=include_dirs,
        language="c++",
    )
]


setup(cmdclass={"build_ext": BuildExt}, ext_modules=ext_modules)
