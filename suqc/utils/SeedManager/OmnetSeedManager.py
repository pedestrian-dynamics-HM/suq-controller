import random
import copy
from typing import Dict, Any, List, Union

from suqc.utils.SeedManager.SeedManager import SeedManager


class OmnetSeedManager(SeedManager):
    def __init__(self, par_variations: List[Dict[str, Any]],
                 rep_count: int = 1,
                 omnet_fixed: bool = True,
                 vadere_fixed: Union[bool, None] = True,
                 seed = 10):
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
        super().__init__(par_variations=par_variations, rep_count=rep_count, seed= 10)
        self.omnet_seed_range = range(1, 255)
        self.vadere_seed_range = range(1, 100000)
        self.omnet_fixed = omnet_fixed
        self.vadere_fixed = vadere_fixed
        if rep_count == 0:
            raise ValueError("rep_count of 0 is not supported")

    def _set_seed(self, variation: Dict[str, Any], vadere_seed: int, omnet_seed: int):
        if "vadere" not in variation:
            variation["vadere"] = {}
        if "omnet" not in variation:
            variation["omnet"] = {}

        # vadere seed
        if self.vadere_fixed is None:
            pass # vadere is not used
        elif self.vadere_fixed:
            self._add_vadere_seed_fixed(variation)
        else:
            self._add_vadere_seed_random(variation, vadere_seed)

        # omnet seed
        if self.omnet_fixed:
            self._add_omnet_seed_fixed(variation)
        else:
            self._add_omnet_seed_random(variation, omnet_seed)

    def _add_vadere_seed_fixed(self, parameter_variations: Dict[str, Any]) -> "OmnetSeedManager":
        # use fixed seed defined in scenario file
        parameter_variations["omnet"]["*.traci.launcher.useVadereSeed"] = "true"
        parameter_variations["vadere"]["attributesSimulation.useFixedSeed"] = True
        return self

    def _add_vadere_seed_random(self, parameter_variations: Dict[str, Any], seed: int) -> "OmnetSeedManager":
        # use random seed in vadere provided from omnet ini file
        parameter_variations["omnet"]["*.traci.launcher.useVadereSeed"] = "false"
        parameter_variations["omnet"]["*.traci.launcher.seed"] = str(seed)
        return self

    def _add_omnet_seed_fixed(self, parameter_variations: Dict[str, Any]):
        # use fixed seed defined in omnet ini file
        pass

    def _add_omnet_seed_random(self, parameter_variations: Dict[str, Any], seed: int) -> "OmnetSeedManager":
        # use random seed for omnet
        parameter_variations["omnet"]["seed-set"] = str(seed)
        return self

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
        vadere_samples = random.sample(self.vadere_seed_range, self.repetition_count)
        omnet_samples = random.sample(self.omnet_seed_range, self.repetition_count)
        for parameter_variation in self.parameter_variations:
            for rep in range(self.repetition_count):
                copied_element = copy.deepcopy(parameter_variation)
                self._set_seed(variation=copied_element, omnet_seed=omnet_samples[rep], vadere_seed=vadere_samples[rep])
                ret.append(copied_element)
        return ret
