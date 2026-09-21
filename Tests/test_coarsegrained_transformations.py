"""Regression tests for spreading coarse-grained chain representations."""

import logging
import unittest

import numpy as np
from Bio.PDB import Atom, Chain, Model, Residue, Structure
from scipy.spatial.transform import Rotation

from pyGrained import CoarseGrainedBase


def make_structure(structure_id, chains):
    structure = Structure.Structure(structure_id)
    model = Model.Model(0)
    structure.add(model)

    serial = 1
    for chain_id, positions in chains.items():
        chain = Chain.Chain(chain_id)
        model.add(chain)
        for residue_id, position in enumerate(positions, start=1):
            residue = Residue.Residue((" ", residue_id, " "), "ALA", " ")
            atom = Atom.Atom(
                "CA", np.asarray(position, dtype=float), 0.0, 1.0, " ",
                " CA ", serial, element="C",
            )
            atom.set_charge(0.0)
            atom.mass = 12.0
            atom.radius = 1.0
            residue.add(atom)
            chain.add(residue)
            serial += 1

    return structure


class CoarseGrainedTransformationTests(unittest.TestCase):
    def setUp(self):
        self.base = object.__new__(CoarseGrainedBase)
        self.base.logger = logging.getLogger("pyGrained.tests")

    def spread_beads(self, reference_ca, mobile_ca, bead_positions):
        input_structure = make_structure(
            "all_atom", {"A": reference_ca, "B": mobile_ca}
        )
        aggregated_ca = make_structure("aggregated", {"A": reference_ca})
        classes = {"A": {"leader": "A", "members": ["A", "B"]}}

        transformations = self.base._CoarseGrainedBase__computeTransformations(
            input_structure, aggregated_ca, classes
        )
        classes["A"]["transformations"] = transformations["A"]

        coarse_grained = make_structure("coarse_grained", {"A": bead_positions})
        return self.base._CoarseGrainedBase__spreadStructure(
            coarse_grained, classes
        )

    @staticmethod
    def chain_positions(structure, chain_id):
        chain = next(chain for chain in structure.get_chains()
                     if chain.get_id() == chain_id)
        return np.array([atom.get_coord() for atom in chain.get_atoms()])

    def test_spreading_uses_ca_transform_for_different_bead_centroid(self):
        reference_ca = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ])
        rotation = Rotation.from_euler("z", 90, degrees=True)
        translation = np.array([7.0, -3.0, 2.0])
        mobile_ca = rotation.apply(reference_ca) + translation

        # Deliberately choose a bead centroid different from the CA centroid.
        bead_positions = np.array([
            [10.0, 1.0, -2.0],
            [12.0, 4.0, 5.0],
        ])
        spread = self.spread_beads(reference_ca, mobile_ca, bead_positions)

        actual = self.chain_positions(spread, "B")
        expected = rotation.apply(bead_positions) + translation

        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(
            self.chain_positions(spread, "A"), bead_positions, atol=1e-12
        )

    def test_spreading_applies_pure_translation(self):
        reference_ca = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ])
        translation = np.array([-4.0, 6.0, 1.5])
        bead_positions = np.array([
            [10.0, 1.0, -2.0],
            [12.0, 4.0, 5.0],
        ])
        spread = self.spread_beads(
            reference_ca, reference_ca + translation, bead_positions
        )

        actual = self.chain_positions(spread, "B")
        np.testing.assert_allclose(
            actual, bead_positions + translation, atol=1e-12
        )

if __name__ == "__main__":
    unittest.main()
