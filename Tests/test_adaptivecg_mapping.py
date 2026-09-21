"""Numerical regression tests for the AdaptiveCG bead mapping."""

import logging
import unittest

import numpy as np
from scipy.spatial import cKDTree

from pyGrained.models.AdaptiveCG import AdaptiveCG


def buildMapper():
    """An AdaptiveCG instance reduced to what _mapping needs (a logger)."""
    mapper = object.__new__(AdaptiveCG)
    mapper.logger = logging.getLogger("pyGrained")
    return mapper


def randomBlobs(nBlobs=6, perBlob=200, seed=0):
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-40.0, 40.0, size=(nBlobs, 3))
    positions = np.repeat(centers, perBlob, axis=0) + rng.normal(0.0, 2.0, (nBlobs*perBlob, 3))
    masses = rng.uniform(1.0, 16.0, nBlobs*perBlob)
    return positions, masses


class AdaptiveCGMappingTests(unittest.TestCase):

    def test_bead_count_follows_resolution(self):
        positions, masses = randomBlobs()
        beads = buildMapper()._mapping(positions, masses, resolution=100, minBeads=1, seed=0)
        self.assertEqual(len(beads), int(positions.shape[0]/100)+1)
        self.assertEqual(beads.shape[1], 3)
        self.assertTrue(np.isfinite(beads).all())

    def test_chain_is_dropped_below_minBeads(self):
        positions, masses = randomBlobs()
        # resolution coarse enough that Nbeads == 2, which minBeads rejects
        resolution = positions.shape[0]
        self.assertEqual(
            buildMapper()._mapping(positions, masses, resolution, minBeads=2, seed=0), []
        )

    def test_mapping_is_a_centroidal_voronoi_tessellation(self):
        """The paper's mapping in the sigma -> 0 limit: every bead sits at the
        mass-weighted centroid of its own Voronoi cell (Eq. 8 with chi the
        characteristic function of the cell)."""
        positions, masses = randomBlobs()
        beads = buildMapper()._mapping(positions, masses, resolution=100, minBeads=1, seed=0)

        labels = cKDTree(beads).query(positions)[1]
        self.assertEqual(len(set(labels)), len(beads), "every bead must own atoms")

        for mu, bead in enumerate(beads):
            cell = labels == mu
            centroid = np.average(positions[cell], axis=0, weights=masses[cell])
            np.testing.assert_allclose(bead, centroid, atol=1e-3)

    def test_mapping_conserves_total_mass(self):
        positions, masses = randomBlobs()
        beads = buildMapper()._mapping(positions, masses, resolution=100, minBeads=1, seed=0)

        labels = cKDTree(beads).query(positions)[1]
        beadMasses = np.bincount(labels, weights=masses, minlength=len(beads))
        self.assertAlmostEqual(beadMasses.sum(), masses.sum(), places=6)

    def test_mapping_is_reproducible_for_a_fixed_seed(self):
        positions, masses = randomBlobs()
        first  = buildMapper()._mapping(positions, masses, resolution=100, minBeads=1, seed=7)
        second = buildMapper()._mapping(positions, masses, resolution=100, minBeads=1, seed=7)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
