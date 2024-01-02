import lastmile_utils.lib.core.api as core_utils


def test_deprefix():
    s = "the quick brown fox"
    pfx = "the "
    assert core_utils.deprefix(s, pfx) == "quick brown fox"
