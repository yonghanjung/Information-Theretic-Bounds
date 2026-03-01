import importlib


def test_import_itbound_exports():
    itbound = importlib.import_module("itbound")
    assert hasattr(itbound, "compute_causal_bounds")
