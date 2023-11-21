import lastmile_utils.lib.core.api as cu

def test_deprefix():
    s = "the quick brown fox"
    pfx = "the "
    assert cu.deprefix(s, pfx) == "quick brown fox"