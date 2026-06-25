# Copyright 2026 WOTA CORP.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for parameter transforms."""

import numpy as np
import numpy.testing as npt
import pytest

from minibayes.transforms import (
    AffineTransform,
    CorrCholeskyTransform,
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


def numerical_jacobian_cholesky(
    transform: CorrCholeskyTransform, chol: np.ndarray, eps: float = 1e-5
) -> float:
    """
    Compute numerical log Jacobian for CorrCholeskyTransform via finite differences.

    Uses central differences on each unconstrained element to estimate
    the log absolute determinant of the Jacobian of the inverse transform.
    """
    y = transform.forward(chol)
    n_params = len(y)
    log_det = 0.0

    for i in range(n_params):
        y_plus = y.copy()
        y_minus = y.copy()
        y_plus[i] += eps
        y_minus[i] -= eps

        chol_plus = transform.inverse(y_plus)
        chol_minus = transform.inverse(y_minus)

        # Compute derivative of each Cholesky element w.r.t. y[i]
        d_chol = (chol_plus - chol_minus) / (2 * eps)
        # Extract off-diagonal elements in same order as forward transform
        dim = transform._dim
        idx = 0
        for row in range(1, dim):
            for col in range(row):
                if idx == i:
                    # This is the diagonal element of the Jacobian
                    log_det += float(np.log(np.abs(d_chol[row, col])))
                idx += 1

    return log_det


def make_random_cholesky(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random valid Cholesky factor for a correlation matrix."""
    # Generate random correlations via the onion method
    chol = np.eye(dim)
    for i in range(1, dim):
        # Generate random partial correlations in (-1, 1)
        remaining = 1.0
        for j in range(i):
            # Random value in valid range
            max_val = np.sqrt(remaining) * 0.9  # Stay away from boundaries
            chol[i, j] = rng.uniform(-max_val, max_val)
            remaining -= chol[i, j] ** 2
        chol[i, i] = np.sqrt(max(remaining, 1e-10))
    return chol


class TestCorrCholeskyTransform:
    """Tests for CorrCholeskyTransform."""

    @pytest.mark.parametrize("dim", [2, 3, 4, 5])
    def test_forward_inverse_roundtrip(self, dim: int) -> None:
        """Test inverse(forward(L)) == L for various dimensions."""
        rng = np.random.default_rng(42)
        t = CorrCholeskyTransform(dim=dim)
        chol = make_random_cholesky(dim, rng)

        y = t.forward(chol)
        chol_recovered = t.inverse(y)
        npt.assert_allclose(chol_recovered, chol, rtol=1e-6, atol=1e-10)

    def test_forward_output_shape(self) -> None:
        """Test forward returns correct shape (n_offdiag,)."""
        for dim in [2, 3, 4, 5]:
            t = CorrCholeskyTransform(dim=dim)
            chol = np.eye(dim)  # Identity Cholesky
            y = t.forward(chol)
            expected_size = dim * (dim - 1) // 2
            assert y.shape == (expected_size,), f"dim={dim}"

    def test_inverse_output_shape(self) -> None:
        """Test inverse returns correct shape (dim, dim)."""
        for dim in [2, 3, 4, 5]:
            t = CorrCholeskyTransform(dim=dim)
            n_params = dim * (dim - 1) // 2
            y = np.zeros(n_params)  # All zeros -> identity correlation
            chol = t.inverse(y)
            assert chol.shape == (dim, dim), f"dim={dim}"

    def test_inverse_produces_lower_triangular(self) -> None:
        """Test inverse output is lower triangular."""
        rng = np.random.default_rng(123)
        for dim in [2, 3, 4]:
            t = CorrCholeskyTransform(dim=dim)
            n_params = dim * (dim - 1) // 2
            y = rng.standard_normal(n_params)
            chol = t.inverse(y)

            # Upper triangle (excluding diagonal) should be zero
            for i in range(dim):
                for j in range(i + 1, dim):
                    assert chol[i, j] == 0.0, f"dim={dim}, i={i}, j={j}"

    def test_inverse_produces_unit_row_norms(self) -> None:
        """Test each row of inverse(y) has unit norm (correlation property)."""
        rng = np.random.default_rng(456)
        for dim in [2, 3, 4, 5]:
            t = CorrCholeskyTransform(dim=dim)
            n_params = dim * (dim - 1) // 2
            y = rng.standard_normal(n_params)
            chol = t.inverse(y)

            for i in range(dim):
                row_norm = np.linalg.norm(chol[i, : i + 1])
                npt.assert_allclose(row_norm, 1.0, rtol=1e-6, atol=1e-10)

    def test_inverse_produces_valid_correlation_matrix(self) -> None:
        """Test L @ L.T is a valid correlation matrix."""
        rng = np.random.default_rng(789)
        for dim in [2, 3, 4]:
            t = CorrCholeskyTransform(dim=dim)
            n_params = dim * (dim - 1) // 2
            y = rng.standard_normal(n_params)
            chol = t.inverse(y)
            corr = chol @ chol.T

            # Diagonal should be 1
            npt.assert_allclose(np.diag(corr), np.ones(dim), rtol=1e-6)

            # Off-diagonal should be in (-1, 1)
            for i in range(dim):
                for j in range(i + 1, dim):
                    assert -1 < corr[i, j] < 1, f"dim={dim}, corr[{i},{j}]={corr[i,j]}"

            # Should be positive semi-definite
            eigvals = np.linalg.eigvalsh(corr)
            assert np.all(eigvals >= -1e-10), f"dim={dim}, eigvals={eigvals}"

    def test_identity_correlation_forward(self) -> None:
        """Test forward of identity Cholesky gives zeros."""
        for dim in [2, 3, 4]:
            t = CorrCholeskyTransform(dim=dim)
            chol = np.eye(dim)
            y = t.forward(chol)
            npt.assert_allclose(y, np.zeros_like(y), atol=1e-10)

    def test_identity_correlation_inverse(self) -> None:
        """Test inverse of zeros gives identity Cholesky."""
        for dim in [2, 3, 4]:
            t = CorrCholeskyTransform(dim=dim)
            n_params = dim * (dim - 1) // 2
            y = np.zeros(n_params)
            chol = t.inverse(y)
            npt.assert_allclose(chol, np.eye(dim), atol=1e-10)

    def test_extreme_positive_correlation_dim2(self) -> None:
        """Test near +1 correlation for dim=2."""
        t = CorrCholeskyTransform(dim=2)
        # Cholesky for correlation 0.99: [[1, 0], [0.99, sqrt(1-0.99^2)]]
        rho = 0.99
        chol = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho**2)]])
        y = t.forward(chol)
        chol_recovered = t.inverse(y)
        npt.assert_allclose(chol_recovered, chol, rtol=1e-4)

    def test_extreme_negative_correlation_dim2(self) -> None:
        """Test near -1 correlation for dim=2."""
        t = CorrCholeskyTransform(dim=2)
        # Cholesky for correlation -0.99
        rho = -0.99
        chol = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho**2)]])
        y = t.forward(chol)
        chol_recovered = t.inverse(y)
        npt.assert_allclose(chol_recovered, chol, rtol=1e-4)

    def test_log_det_jacobian_identity(self) -> None:
        """Test Jacobian at identity Cholesky."""
        for dim in [2, 3, 4]:
            t = CorrCholeskyTransform(dim=dim)
            chol = np.eye(dim)
            jac = t.log_det_jacobian(chol)
            # Should be finite
            assert np.isfinite(jac), f"dim={dim}, jac={jac}"
            # At identity, all z values are 0, so log(1-z^2) = 0
            # and remaining = 1, so 0.5*log(remaining) = 0
            npt.assert_allclose(float(jac), 0.0, atol=1e-10)

    def test_log_det_jacobian_finite(self) -> None:
        """Test Jacobian is finite for valid Cholesky factors."""
        rng = np.random.default_rng(111)
        for dim in [2, 3, 4]:
            t = CorrCholeskyTransform(dim=dim)
            chol = make_random_cholesky(dim, rng)
            jac = t.log_det_jacobian(chol)
            assert np.isfinite(jac), f"dim={dim}, jac={jac}"

    @pytest.mark.parametrize("dim", [2, 3])
    def test_jacobian_numerical(self, dim: int) -> None:
        """Verify Jacobian via numerical differentiation."""
        rng = np.random.default_rng(222 + dim)
        t = CorrCholeskyTransform(dim=dim)
        chol = make_random_cholesky(dim, rng)

        analytical = float(t.log_det_jacobian(chol))
        numerical = numerical_jacobian_cholesky(t, chol, eps=1e-5)

        npt.assert_allclose(analytical, numerical, rtol=0.05, atol=0.1)

    def test_dim2_specific_values(self) -> None:
        """Test specific known values for dim=2."""
        t = CorrCholeskyTransform(dim=2)

        # For dim=2, Cholesky is [[1, 0], [rho, sqrt(1-rho^2)]]
        # forward should give arctanh(rho)
        for rho in [-0.5, 0.0, 0.3, 0.7]:
            chol = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho**2)]])
            y = t.forward(chol)
            expected = np.arctanh(rho)
            npt.assert_allclose(y[0], expected, rtol=1e-6)

    def test_dim2_inverse_specific_values(self) -> None:
        """Test inverse for specific values for dim=2."""
        t = CorrCholeskyTransform(dim=2)

        # inverse(arctanh(rho)) should give [[1, 0], [rho, sqrt(1-rho^2)]]
        for rho in [-0.5, 0.0, 0.3, 0.7]:
            y = np.array([np.arctanh(rho)])
            chol = t.inverse(y)
            expected = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho**2)]])
            npt.assert_allclose(chol, expected, rtol=1e-6)
