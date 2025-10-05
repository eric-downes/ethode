"""Pytest configuration and shared test utilities."""

import pytest
import numpy as np
import jax.numpy as jnp
from typing import Union


# Default tolerances for float comparisons
# These handle float32 precision while remaining strict enough to catch real bugs
RTOL_DEFAULT = 1e-6  # Relative tolerance
ATOL_DEFAULT = 1e-7  # Absolute tolerance


def assert_close(
    actual: Union[float, jnp.ndarray, np.ndarray],
    expected: Union[float, jnp.ndarray, np.ndarray],
    rtol: float = RTOL_DEFAULT,
    atol: float = ATOL_DEFAULT,
    msg: str = ""
):
    """Assert that two values are close within tolerance.

    Handles JAX arrays, NumPy arrays, and Python floats uniformly.
    Uses sensible defaults for float32 precision.

    Args:
        actual: Actual value
        expected: Expected value
        rtol: Relative tolerance (default: 1e-6)
        atol: Absolute tolerance (default: 1e-7)
        msg: Optional message for assertion failure

    Example:
        >>> state = HawkesState.initialize(0.01)
        >>> assert_close(state.current_intensity, 0.01)
    """
    # Convert to float for comparison
    actual_val = float(actual) if hasattr(actual, '__float__') else actual
    expected_val = float(expected) if hasattr(expected, '__float__') else expected

    # Use pytest.approx for nice error messages
    assert actual_val == pytest.approx(expected_val, rel=rtol, abs=atol), (
        f"{msg}\nExpected: {expected_val}\nActual: {actual_val}\n"
        f"Diff: {abs(actual_val - expected_val)}"
    )


def assert_array_close(
    actual: Union[jnp.ndarray, np.ndarray],
    expected: Union[jnp.ndarray, np.ndarray],
    rtol: float = RTOL_DEFAULT,
    atol: float = ATOL_DEFAULT,
    msg: str = ""
):
    """Assert that two arrays are close within tolerance.

    Uses numpy.testing.assert_allclose for detailed error messages.

    Args:
        actual: Actual array
        expected: Expected array
        rtol: Relative tolerance (default: 1e-6)
        atol: Absolute tolerance (default: 1e-7)
        msg: Optional message for assertion failure

    Example:
        >>> outputs = jnp.array([1.0, 0.5, 0.2])
        >>> expected = jnp.array([1.0, 0.5, 0.2])
        >>> assert_array_close(outputs, expected)
    """
    # Convert JAX arrays to numpy for comparison
    actual_np = np.asarray(actual)
    expected_np = np.asarray(expected)

    np.testing.assert_allclose(
        actual_np, expected_np,
        rtol=rtol, atol=atol,
        err_msg=msg
    )


@pytest.fixture
def close():
    """Fixture providing assert_close function.

    Usage:
        def test_something(close):
            close(actual, expected)
    """
    return assert_close


@pytest.fixture
def array_close():
    """Fixture providing assert_array_close function.

    Usage:
        def test_something(array_close):
            array_close(actual_array, expected_array)
    """
    return assert_array_close
