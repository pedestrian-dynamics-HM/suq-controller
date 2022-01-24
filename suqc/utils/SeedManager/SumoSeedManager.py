from typing import List, Dict, Union, Any
import random
import copy
from typing import Dict, Any, List

from suqc.utils.SeedManager.SeedManager import SeedManager


class SumoSeedManager(SeedManager):
    def __init__(self, par_variations: List[Dict[str, Any]], rep_count: int = 1, omnet_fixed: bool = True, sumo_fixed:bool =True, seed=10):
        """SeedManager class for crownet based simulation

        Attributes:
        -----
        par_variations: List[Dict[str, Any]]
            the parameter variation that needs to be seeded
        seed_config:  Dict[str, str]
            seed configuration (default {"vadere": "random", "omnet": "random"})
        rep_count: int
            repetition count determines how many seed different seed configuration are set per variation
        """
        super().__init__(par_variations=par_variations, rep_count=rep_count, seed=seed)
        self.omnet_seed_range = range(1, 255)
        self.omnet_fixed = omnet_fixed
        self.sumo_fixed = sumo_fixed
        if rep_count == 0:
            raise ValueError("rep_count of 0 is not supported")

    def _add_omnet_seed_random(self, parameter_variations: Dict[str, Any], seed: int) -> "SumoSeedManager":
        # use random seed for omnet
        parameter_variations["omnet"]["seed-set"] = str(seed)
        return self

    def _set_seed(self, variation: Dict[str, Any], omnet_seed: int):
        # todo: if sumo gets implemented, add new sumo seed

        # omnet seed
        if self.omnet_fixed:
            pass
        else:
            # seeds = [str(random.randint(1, 255)) for _ in range(self.points.shape[0])]
            # self.points.insert(0, ("Parameter", "omnet", "seed-set"), seeds, True)
            self._add_omnet_seed_random(variation, omnet_seed)

    def get_new_seed_variation(self) -> List[
        Dict[str, Any]]:
        """
                Generates a new seed variation.

                Returns
                -------
                out : List[Dict[str, Any]]
                    List of seeded variants.

                Examples
                --------
                >>> OmnetSeedManager(par_variations=[variation_1, variation_2], rep_count=2).get_new_seed_variation()
                [variation_1_seed_1,
                variation_1_seed_2,
                variation_2_seed_1,
                variation_2_seed_2]
                >>> OmnetSeedManager(par_variations=[variation_1], rep_count=4).get_new_seed_variation()
                [variation_1_seed_1,
                variation_1_seed_2,
                variation_1_seed_3,
                variation_1_seed_4]
                >>> OmnetSeedManager(par_variations=[variation_1, variation_2]).get_new_seed_variation()
                [variation_1_seed_1,
                variation_1_seed_2]

        """
        ret: List[Dict[str, Any]] = []
        omnet_samples = random.sample(self.omnet_seed_range, self.repetition_count)
        for parameter_variation in self.parameter_variations:
            for rep in range(self.repetition_count):
                copied_element = copy.deepcopy(parameter_variation)
                self._set_seed(variation=copied_element, omnet_seed=omnet_samples[rep])
                ret.append(copied_element)
        return ret
