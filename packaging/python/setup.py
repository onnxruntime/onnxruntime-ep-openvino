# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: MIT

import sysconfig
from setuptools import setup
from wheel.bdist_wheel import bdist_wheel


class PlatformWheel(bdist_wheel):
    """Force a py3-none-<platform> tag for a data-only (no ext_modules) wheel."""

    def get_tag(self):
        plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        return "py3", "none", plat


setup(cmdclass={"bdist_wheel": PlatformWheel})
