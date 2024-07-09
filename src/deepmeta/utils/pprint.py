"""
Pretty print file
"""
from typing import Any

def print_red(skk: Any):
    """
    Print in red.

    :param skk: Str to print
    :type skk: str
    """
    print("\033[91m{}\033[00m".format(skk))

def print_gre(skk: Any):
    """
    Print in green.

    :param skk: Str to print
    :type skk: str
    """
    print("\033[92m{}\033[00m".format(skk))


def print_yel(skk: Any):
    """
    Print in yellow.

    :param skk: Str to print
    :type skk: str
    """
    print("\033[93m{}\033[00m".format(skk))


def print_mag(skk: Any):
    """
    Print in magenta.

    :param skk: Str to print
    :type skk: str
    """
    print("\033[95m{}\033[00m".format(skk))


def print_bold_red(skk: Any):
    """
    Print in bold and red.

    :param skk: Str to print
    :type skk: str
    """
    print("\033[1m\033[91m{}\033[00m".format(skk))


def print_bold_green(skk: Any):
    """
    Print in bold and green.

    :param skk: Str to print
    :type skk: str
    """
    print("\033[1m\033[92m{}\033[00m".format(skk))
