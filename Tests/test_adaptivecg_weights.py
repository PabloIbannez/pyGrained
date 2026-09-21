"""Numerical regression tests for the AdaptiveCG mapping weights."""

import unittest

import numpy as np

from pyGrained.models.AdaptiveCG import ChainAdaptiveCG


class AdaptiveCGWeightsTests(unittest.TestCase):
    def test_single_bead_survives_gaussian_underflow(self):
        coords = np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        model = ChainAdaptiveCG(
            1, coords, np.ones(2), R_init=np.zeros((1, 3)), sigma=0.2
        )

        np.testing.assert_array_equal(model.compute_chi(), np.ones((2, 1)))
        centers, _ = model.optimize(max_iter=2)
        self.assertTrue(np.isfinite(centers).all())
        np.testing.assert_allclose(centers, np.zeros((1, 3)))

    def test_narrow_gaussians_normalize_distant_and_tied_atoms(self):
        coords = np.array([[10.0, 0.0, 0.0], [100.0, 0.0, 0.0],
                           [-100.0, 0.0, 0.0]])
        centers = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        model = ChainAdaptiveCG(
            2, coords, np.ones(3), R_init=centers, sigma=0.2
        )

        chi = model.compute_chi()
        self.assertTrue(np.isfinite(chi).all())
        self.assertTrue((chi >= 0.0).all())
        np.testing.assert_allclose(chi.sum(axis=1), np.ones(3))
        np.testing.assert_allclose(chi, [[0.5, 0.5], [0.0, 1.0], [1.0, 0.0]])

    def test_matches_original_formula_when_gaussians_are_representable(self):
        coords = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0],
                           [3.0, 0.0, 1.0]])
        centers = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        masses = np.array([1.0, 2.0, 3.0])
        sigma = 1.5
        dist2 = np.sum((coords[:, None, :] - centers[None, :, :])**2, axis=2)
        weights = np.exp(-dist2 / (2 * sigma**2))
        expected = weights / weights.sum(axis=1, keepdims=True)
        model = ChainAdaptiveCG(2, coords, masses, R_init=centers, sigma=sigma)

        chi = model.compute_chi()
        np.testing.assert_allclose(chi, expected, rtol=1e-13, atol=1e-15)
        effective_masses = (masses[:, None] * chi).sum(axis=0)
        np.testing.assert_allclose(effective_masses.sum(), masses.sum())
        updated = model.update_R(chi)
        np.testing.assert_allclose(
            np.average(updated, axis=0, weights=effective_masses),
            np.average(coords, axis=0, weights=masses),
        )

    def test_rejects_nonpositive_or_nonfinite_sigma(self):
        for sigma in (0.0, -0.2, np.nan, np.inf, -np.inf):
            with self.subTest(sigma=sigma):
                with self.assertRaisesRegex(ValueError, "sigma.*finite.*positive"):
                    ChainAdaptiveCG(
                        1, np.zeros((1, 3)), np.ones(1),
                        R_init=np.zeros((1, 3)), sigma=sigma,
                    )

    def test_rejects_bead_with_zero_effective_mass(self):
        model = ChainAdaptiveCG(
            2, np.zeros((1, 3)), np.ones(1),
            R_init=np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]), sigma=0.2,
        )

        with self.assertRaisesRegex(ValueError, r"effective mass.*\[1\]"):
            model.optimize(max_iter=1)

    def test_rejects_nonfinite_effective_mass(self):
        for mass in (np.nan, np.inf):
            with self.subTest(mass=mass):
                model = ChainAdaptiveCG(
                    1, np.ones((1, 3)), np.array([mass]),
                    R_init=np.ones((1, 3)), sigma=0.2,
                )
                with self.assertRaisesRegex(ValueError, r"effective mass.*\[0\]"):
                    model.update_R(model.compute_chi())


if __name__ == "__main__":
    unittest.main()
