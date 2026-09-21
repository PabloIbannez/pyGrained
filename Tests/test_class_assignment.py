"""Regression tests for the grouping of chains into classes.

The leader of a class decides which chain is actually coarse-grained, and the
result is spread to every other member. A class of chains that share a sequence
but not a conformation, which is what a capsid is made of, therefore gives a
different model for a different leader. The leader has to be reproducible.
"""

import logging
import os
import subprocess
import sys
import unittest

from Bio.PDB import Structure, Model, Chain, Residue

from pyGrained import CoarseGrainedBase


def buildGrouper():
    """A CoarseGrainedBase reduced to what __getClasses needs (a logger)."""
    grouper = object.__new__(CoarseGrainedBase)
    grouper.logger = logging.getLogger("pyGrained")
    return grouper


def buildStructure(chainSequences):
    """One model holding a chain per entry of {chainId: one letter sequence}."""

    threeLetter = {"A":"ALA","C":"CYS","G":"GLY","S":"SER","V":"VAL"}

    structure = Structure.Structure("test")
    model     = Model.Model(0)
    structure.add(model)

    for chainId, sequence in chainSequences.items():
        chain = Chain.Chain(chainId)
        model.add(chain)
        # getChainSequence stops one short of the last residue, so the
        # sequence is padded to keep these fixtures readable.
        for i, letter in enumerate(sequence + "G", start=1):
            residue = Residue.Residue((" ", i, " "), threeLetter[letter], "")
            chain.add(residue)

    return structure


LEADER_OF_TWELVE_EQUAL_CHAINS = f"""
import logging, sys
logging.getLogger("pyGrained").setLevel(logging.CRITICAL)
sys.path.insert(0,{os.path.dirname(os.path.abspath(__file__))!r})
from test_class_assignment import buildGrouper, buildStructure
classes = buildGrouper()._CoarseGrainedBase__getClasses(
    buildStructure({{ch:"ACGSV" for ch in "ABCDEFGHIJKL"}}))
print(list(classes.values())[0]["leader"])
"""


class ClassAssignmentTests(unittest.TestCase):

    def test_leader_is_the_shortest_member(self):
        classes = buildGrouper()._CoarseGrainedBase__getClasses(
            buildStructure({"A":"ACGSV", "B":"CGS"}))

        self.assertEqual(len(classes), 1)
        info = list(classes.values())[0]
        self.assertEqual(info["leader"], "B")
        self.assertEqual(info["members"], ["A","B"])

    def test_chains_of_unrelated_sequence_are_separate_classes(self):
        classes = buildGrouper()._CoarseGrainedBase__getClasses(
            buildStructure({"A":"ACAC", "B":"SVSV"}))

        self.assertEqual(len(classes), 2)
        for info in classes.values():
            self.assertEqual(len(info["members"]), 1)

    def test_leader_of_equal_chains_does_not_depend_on_hash_seed(self):
        """The members are accumulated in a set, and a set of strings iterates
        in an order that depends on the interpreter's hash seed. The seed is
        fixed once per process, so this has to be checked across processes:
        before the members were sorted, twelve seeds gave eight leaders."""

        leaders = set()
        for seed in range(12):
            env = dict(os.environ, PYTHONHASHSEED=str(seed))
            run = subprocess.run([sys.executable,"-c",LEADER_OF_TWELVE_EQUAL_CHAINS],
                                 env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            leaders.add(run.stdout.strip())

        self.assertEqual(leaders, {"A"})


if __name__ == "__main__":
    unittest.main()
