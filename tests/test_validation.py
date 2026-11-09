"""Input contract at the API boundary: windows, domain IDs, and partial labels."""

import numpy as np
import pytest

from neuro_rave._validation import as_windows, check_layout, encode_categories


@pytest.mark.parametrize("shape", [(4, 10), (4, 10, 3)])
def test_as_windows_accepts_both_ranks_and_converts_to_float32(shape):
    X = np.random.default_rng(0).normal(size=shape)  # float64
    windows = as_windows(X)
    assert windows.dtype == np.float32
    assert windows.shape == shape


def test_as_windows_accepts_integers():
    assert as_windows(np.arange(6).reshape(2, 3)).dtype == np.float32


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((2, 3), dtype=complex),
        np.ones((2, 3), dtype=bool),
        np.array([["a", "b"], ["c", "d"]]),
        np.array([[1.0, None]], dtype=object),
    ],
)
def test_as_windows_rejects_non_real_dtypes(bad):
    with pytest.raises(TypeError, match="real numbers"):
        as_windows(bad)


@pytest.mark.parametrize("shape", [(5,), (2, 3, 4, 5)])
def test_as_windows_rejects_wrong_rank_and_reports_shape(shape):
    with pytest.raises(ValueError, match=r"got shape \(" + ", ".join(map(str, shape))):
        as_windows(np.zeros(shape))


@pytest.mark.parametrize("value", [np.nan, np.inf, 1e300])
def test_as_windows_rejects_non_finite_after_conversion(value):
    X = np.zeros((3, 4))
    X[1, 2] = value
    with pytest.raises(ValueError, match="1 non-finite"):
        as_windows(X)


def test_check_layout_reports_expected_and_actual():
    with pytest.raises(ValueError, match=r"\(5, 8, 2\).*\(samples, 8, 3\)"):
        check_layout(np.zeros((5, 8, 2), dtype=np.float32), (8, 3))


def test_string_categories_are_sorted_and_coded():
    enc = encode_categories(["s2", "s1", "s2"], name="domains", n_rows=3, allow_missing=False)
    assert list(enc.categories) == ["s1", "s2"]
    assert enc.codes.tolist() == [1, 0, 1]
    assert enc.known.all()


def test_nonconsecutive_int_labels_with_missing():
    enc = encode_categories([10, None, 3, 7, None], name="y", n_rows=5, allow_missing=True)
    assert enc.categories.tolist() == [3, 7, 10]
    assert enc.codes.tolist() == [2, -1, 0, 1, -1]
    assert enc.known.tolist() == [True, False, True, True, False]


def test_domains_reject_none():
    with pytest.raises(ValueError, match="None"):
        encode_categories(["a", None], name="domains", n_rows=2, allow_missing=False)


def test_nan_label_points_to_none():
    with pytest.raises(ValueError, match="Use None"):
        encode_categories([1.0, np.nan, 2.0], name="y", n_rows=3, allow_missing=True)


def test_mixed_unorderable_categories_rejected():
    with pytest.raises(TypeError, match="cannot be ordered"):
        encode_categories(["a", 1], name="y", n_rows=2, allow_missing=True)


def test_length_mismatch_reports_both_lengths():
    with pytest.raises(ValueError, match="3 entries, but X has 4 rows"):
        encode_categories([0, 1, 0], name="domains", n_rows=4, allow_missing=False)


def test_requires_two_distinct_values():
    with pytest.raises(ValueError, match="at least 2 distinct"):
        encode_categories(["a", "a"], name="domains", n_rows=2, allow_missing=False)
    with pytest.raises(ValueError, match="found 1"):
        encode_categories([None, 5, None], name="y", n_rows=3, allow_missing=True)
