"""Tests for parameter transforms."""

import numpy as np
import numpy.testing as npt
import pytest

from minibayes.transforms import (
    AffineTransform,
    IdentityTransform,
    LogitTransform,
    LogTransform,
)


def numerical_jacobian(transform, x: float, eps: float = 1e-6) -> float:
    """Compute numerical Jacobian via finite differences."""
    phi = transform.forward(np.array([x]))[0]
    theta_plus = transform.inverse(np.array([phi + eps]))[0]
    theta_minus = transform.inverse(np.array([phi - eps]))[0]
    d_theta_d_phi = (theta_plus - theta_minus) / (2 * eps)
    return float(np.log(np.abs(d_theta_d_phi)))


class TestIdentityTransform:
    """Tests for IdentityTransform."""

    def test_forward_inverse_roundtrip(self) -> None:
        """Test inverse(forward(x)) == x."""
        t = IdentityTransform()
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        npt.assert_allclose(t.inverse(t.forward(x)), x)

    def test_log_det_jacobian_zero(self) -> None:
        """Test log_det_jacobian is zero."""
        t = IdentityTransform()
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        npt.assert_allclose(t.log_det_jacobian(x), np.zeros_like(x))

    def test_scalar_input(self) -> None:
        """Test transforms work with scalar inputs."""
        t = IdentityTransform()
        x = 1.5
        result = t.forward(np.array([x]))
        assert result.shape == (1,)
        npt.assert_allclose(t.inverse(result)[0], x)

    @pytest.mark.parametrize("x", [-2.0, 0.0, 2.0])
    def test_jacobian_numerical(self, x: float) -> None:
        """Verify Jacobian via numerical differentiation."""
        t = IdentityTransform()
        analytical = float(t.log_det_jacobian(np.array([x]))[0])
        numerical = numerical_jacobian(t, x)
        npt.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-8)


class TestLogTransform:
    """Tests for LogTransform."""

    def test_forward_inverse_roundtrip(self) -> None:
        """Test inverse(forward(x)) == x."""
        t = LogTransform()
        x = np.array([0.1, 0.5, 1.0, 2.0, 10.0])
        npt.assert_allclose(t.inverse(t.forward(x)), x)

    def test_forward_is_log(self) -> None:
        """Test forward(x) == log(x)."""
        t = LogTransform()
        x = np.array([0.1, 0.5, 1.0, 2.0, 10.0])
        npt.assert_allclose(t.forward(x), np.log(x))

    def test_inverse_is_exp(self) -> None:
        """Test inverse(y) == exp(y)."""
        t = LogTransform()
        y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        npt.assert_allclose(t.inverse(y), np.exp(y))

    def test_log_det_jacobian(self) -> None:
        """Test Jacobian is log(x)."""
        t = LogTransform()
        x = np.array([0.1, 0.5, 1.0, 2.0, 10.0])
        npt.assert_allclose(t.log_det_jacobian(x), np.log(x))

    def test_scalar_input(self) -> None:
        """Test transforms work with scalar inputs."""
        t = LogTransform()
        x = 2.5
        result = t.forward(np.array([x]))
        assert result.shape == (1,)
        npt.assert_allclose(t.inverse(result)[0], x)

    @pytest.mark.parametrize("x", [0.1, 1.0, 10.0])
    def test_jacobian_numerical(self, x: float) -> None:
        """Verify Jacobian via numerical differentiation."""
        t = LogTransform()
        analytical = float(t.log_det_jacobian(np.array([x]))[0])
        numerical = numerical_jacobian(t, x)
        npt.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-8)


class TestLogitTransform:
    """Tests for LogitTransform."""

    def test_forward_inverse_roundtrip(self) -> None:
        """Test inverse(forward(x)) == x."""
        t = LogitTransform()
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        npt.assert_allclose(t.inverse(t.forward(x)), x)

    def test_forward_is_logit(self) -> None:
        """Test forward(x) == log(x/(1-x))."""
        t = LogitTransform()
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        npt.assert_allclose(t.forward(x), np.log(x / (1 - x)))

    def test_inverse_is_sigmoid(self) -> None:
        """Test inverse(y) == 1/(1+exp(-y))."""
        t = LogitTransform()
        y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        npt.assert_allclose(t.inverse(y), 1 / (1 + np.exp(-y)))

    def test_log_det_jacobian(self) -> None:
        """Test Jacobian is log(x) + log(1-x)."""
        t = LogitTransform()
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        npt.assert_allclose(t.log_det_jacobian(x), np.log(x) + np.log(1 - x))

    def test_scalar_input(self) -> None:
        """Test transforms work with scalar inputs."""
        t = LogitTransform()
        x = 0.7
        result = t.forward(np.array([x]))
        assert result.shape == (1,)
        npt.assert_allclose(t.inverse(result)[0], x)

    @pytest.mark.parametrize("x", [0.1, 0.5, 0.9])
    def test_jacobian_numerical(self, x: float) -> None:
        """Verify Jacobian via numerical differentiation."""
        t = LogitTransform()
        analytical = float(t.log_det_jacobian(np.array([x]))[0])
        numerical = numerical_jacobian(t, x)
        npt.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-8)

    def test_edge_values(self) -> None:
        """Test values near boundaries (0, 1)."""
        t = LogitTransform()
        x = np.array([0.001, 0.999])
        # Should not raise, just produce large magnitude outputs
        phi = t.forward(x)
        assert np.all(np.isfinite(phi))
        npt.assert_allclose(t.inverse(phi), x, rtol=1e-5)


class TestAffineTransform:
    """Tests for AffineTransform."""

    def test_forward_inverse_roundtrip(self) -> None:
        """Test inverse(forward(x)) == x."""
        t = AffineTransform(low=-1.0, high=2.0)
        x = np.array([-0.9, 0.0, 0.5, 1.0, 1.9])
        npt.assert_allclose(t.inverse(t.forward(x)), x)

    def test_forward_maps_midpoint_to_zero(self) -> None:
        """Test forward((low+high)/2) == 0."""
        t = AffineTransform(low=0.0, high=1.0)
        npt.assert_allclose(t.forward(np.array([0.5])), np.array([0.0]))

    def test_inverse_maps_zero_to_midpoint(self) -> None:
        """Test inverse(0) == (low+high)/2."""
        t = AffineTransform(low=0.0, high=1.0)
        npt.assert_allclose(t.inverse(np.array([0.0])), np.array([0.5]))

    def test_log_det_jacobian(self) -> None:
        """Test Jacobian is log(x-low) + log(high-x) - log(width)."""
        low, high = -1.0, 2.0
        t = AffineTransform(low=low, high=high)
        x = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
        expected = np.log(x - low) + np.log(high - x) - np.log(high - low)
        npt.assert_allclose(t.log_det_jacobian(x), expected)

    def test_scalar_input(self) -> None:
        """Test transforms work with scalar inputs."""
        t = AffineTransform(low=0.0, high=10.0)
        x = 7.5
        result = t.forward(np.array([x]))
        assert result.shape == (1,)
        npt.assert_allclose(t.inverse(result)[0], x)

    @pytest.mark.parametrize("x", [0.5, 5.0, 9.5])
    def test_jacobian_numerical(self, x: float) -> None:
        """Verify Jacobian via numerical differentiation."""
        t = AffineTransform(low=0.0, high=10.0)
        analytical = float(t.log_det_jacobian(np.array([x]))[0])
        numerical = numerical_jacobian(t, x)
        npt.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-8)

    def test_edge_values(self) -> None:
        """Test values near boundaries."""
        t = AffineTransform(low=0.0, high=1.0)
        x = np.array([0.001, 0.999])
        phi = t.forward(x)
        assert np.all(np.isfinite(phi))
        npt.assert_allclose(t.inverse(phi), x, rtol=1e-5)
