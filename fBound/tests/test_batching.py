import math

import pytest

from utils import choose_batch_size, make_domain_penalty_schedule


def test_choose_batch_size_rule():
    assert choose_batch_size(1) == 16
    assert choose_batch_size(1000) == 16
    assert choose_batch_size(1001) == 32
    assert choose_batch_size(5000) == 32
    assert choose_batch_size(5001) == 64
    assert choose_batch_size(10000) == 64
    assert choose_batch_size(10001) == min(128, int(math.sqrt(10001)))


def test_choose_batch_size_invalid():
    with pytest.raises(ValueError):
        choose_batch_size(0)


def test_domain_penalty_schedule():
    stage1, w_dom = make_domain_penalty_schedule(10, rho=0.3, w1=1e6, w2=1e4)
    assert stage1 == 3
    assert w_dom(0) == 1e6
    assert w_dom(2) == 1e6
    assert w_dom(3) == 1e4
