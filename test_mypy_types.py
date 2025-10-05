#!/usr/bin/env python
"""Test file for mypy type checking with dimension-aware types."""

from ethode.types import (
    TimeScalar,
    PriceScalar,
    to_time_scalar,
    to_price_scalar,
    TimeValue,
    PriceValue,
)
from typing import reveal_type


def process_time(t: TimeScalar) -> TimeScalar:
    """Process a time value."""
    # This should type check correctly
    return TimeScalar(t * 2.0)


def process_price(p: PriceScalar) -> PriceScalar:
    """Process a price value."""
    # This should type check correctly
    return PriceScalar(p * 1.1)


def calculate_rate(distance: float, time: TimeValue) -> float:
    """Calculate rate from distance and time."""
    return distance / time


def main() -> None:
    """Main function demonstrating type usage."""
    # Correct usage
    t1: TimeScalar = to_time_scalar(5.0)
    t2 = process_time(t1)
    print(f"Processed time: {t2}")

    p1: PriceScalar = to_price_scalar(100.0)
    p2 = process_price(p1)
    print(f"Processed price: {p2}")

    # Annotated types
    time_val: TimeValue = 10.0
    rate = calculate_rate(100.0, time_val)
    print(f"Rate: {rate}")

    # This would be a type error if uncommented:
    # wrong = process_time(p1)  # Error: PriceScalar != TimeScalar

    # reveal_type shows inferred types (mypy only)
    if False:  # Only for mypy, not runtime
        reveal_type(t1)  # Should show TimeScalar
        reveal_type(p1)  # Should show PriceScalar


if __name__ == "__main__":
    main()