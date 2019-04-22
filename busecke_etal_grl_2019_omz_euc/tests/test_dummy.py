import pytest
from busecke_etal_grl_2019_omz_euc.dummy import dummy_foo


def test_dummy():
    assert dummy_foo(4) == (4 + 4)
