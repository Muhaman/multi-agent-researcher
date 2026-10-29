import pytest
from pydantic import ValidationError

from researcher.budget import Budget


def test_defaults():
    budget = Budget()
    assert (budget.max_workers, budget.max_searches) == (4, 3)


@pytest.mark.parametrize(
    ("workers", "searches"),
    [(9, 3), (0, 3), (4, 11), (4, 0)],
)
def test_out_of_range_is_rejected(workers: int, searches: int):
    with pytest.raises(ValidationError):
        Budget(max_workers=workers, max_searches=searches)


def test_bounds_are_inclusive():
    assert Budget(max_workers=8, max_searches=10).max_workers == 8
    assert Budget(max_workers=1, max_searches=1).max_searches == 1
