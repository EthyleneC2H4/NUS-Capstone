from __future__ import annotations

import unittest

from scripts.smoke_test import make_args, run_case


class ModelSmokeTests(unittest.TestCase):
    def test_core_model_variants_forward(self):
        run_case("baseline", make_args())
        run_case("heterophily_aware", make_args(heterophily_aware=True))
        run_case("cross_network_attention", make_args(cross_network_attention=True))


if __name__ == "__main__":
    unittest.main()

