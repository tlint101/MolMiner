"""
Scripts to generate molecules de novo
"""
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to reduce error messages during SAFE molecule generation

from typing import Optional, Union, Dict, Any
from rdkit.Chem import BRICS
import pandas as pd
import datamol as dm
from tqdm import tqdm
from mminer._optional import require
from mminer.features.scoring import Score

__all__ = ["SAFEbuild", "BRICSBuild", 'Score']


class SAFEbuild(Score):
    """
    Build molecules using SAFE
    """

    def __init__(self):
        super().__init__()  # call SAScore's __init__
        self.default_model = None
        # todo script to generate future models
        self.custom_model = None

    @classmethod
    def load_default(self, verbose: bool = False, model_dir: Optional[str] = None, device: str = None):
        """
        Load default SAFEDesign model. This will download the model from Hugging Face.
        :param verbose: bool
            Output verbosity
        :param model_dir: Optional[str]
            Optional filepath to model folder to be used instead of default one. Default filepath will be set to
            "~/. cache/huggingface"
        :param device: str
            Optional device where to move the model.
        :return:
        """
        sf = require("safe", "safe", "SAFEbuild")

        # default_model = sf.SAFEDesign.load_default(verbose=verbose)
        default_model = sf.SAFEDesign.load_default(verbose=verbose, model_dir=model_dir, device=device)

        # set default_model to instance variable
        self.default_model = default_model

        return default_model

    def scaffold_decoration(self, scaffold: Union[str, dm.Mol],
                            n_samples_per_trial: int = 10,
                            n_trials: Optional[int] = 1,
                            do_not_fragment_further: Optional[bool] = True,
                            sanitize: bool = False,
                            model: str = 'default',
                            ro5: bool = True,
                            ro3: bool = False,
                            np_score: bool = False,
                            seed: Optional[int] = 42,
                            add_dot: Optional[bool] = True,
                            **kwargs: Optional[Dict[Any, Any]]):
        """
        Generate molecules from an input scaffold.
        :param scaffold: Union[str, dm.Mol]
            Scaffold (Must include attachment points) for decorating.
        :param n_samples_per_trial: int
            Number of molecules to generate for each randomization.
        :param n_trials: Optional[int]
            Set number of randomization to perform.
        :param do_not_fragment_further: Optional[bool]
            Whether to fragment the scaffold further or not.
        :param sanitize: bool
            Whether to sanitize the generated molecules and check if the scaffold is still present.
        :param model: str
            Set the SAFE model. Set to use the default SAFE model.
        :param ro5: bool
            Score generated molecules to Rule of 5 rules.
        :param ro3: bool
            Score generated molecules to Rule of 3 Lead-Like rules. If both ro5 and ro3 are true, then only ro3 will be
            calculated.
        :param np_score: bool
            Generate a Natural Product score for generated molecule.
        :param seed: Optional[int]
            Set the random seed to use. This will set the autoregressive model fo SAFE algorithms and set building
            molecules to be deterministic.
        :param add_dot: Optional[bool]
            Whether to add a dot at the end of the fragments to signal to the model that we want to generate a distinct
            fragment.
        :param kwargs: Optional[Dict[Any, Any]]
        :return:
        """

        """
        There is an issue with using the load_default. To get around it, the function will load the default model 
        internally. This means that a custom model cannot be loaded. This will be troubleshooted and and the functions 
        for custom SAFE models will be wrapped in the future. 
        """
        sf = require("safe", "safe", "SAFEbuild")
        transformers = require("transformers", "safe", "SAFEbuild")

        global safe_model
        if model == 'default':
            safe_model = sf.SAFEDesign.load_default(verbose=True)

        # set safe generation repeatability
        transformers.set_seed(seed)

        # generate smiles
        generated_smiles = safe_model.scaffold_decoration(scaffold=scaffold,
                                                          n_samples_per_trial=n_samples_per_trial,
                                                          n_trials=n_trials,
                                                          do_not_fragment_further=do_not_fragment_further,
                                                          sanitize=sanitize,
                                                          random_seed=seed, add_dot=add_dot,
                                                          **kwargs)

        # # for troubleshooting
        # print("length:", len(generated_smiles))

        # Issue raised if something wrong with generating molecule
        if len(generated_smiles) == 0:
            raise ValueError("ERROR! No Molecules Generated!")

        # remove nonetype from generated_smiles
        generated_smiles = [item for item in generated_smiles if item is not None]

        # output generated smiles and convert into a list of ROMol
        mol_list = [dm.to_mol(x) for x in generated_smiles if isinstance(x, str)]

        # calculate sa and np scores for generated_safe molecules
        data_df = self._sa_score_of_generated_safe(generated_smiles, mol_list, np_score=np_score)

        # set options for ro5 or ro3
        if ro5 is True and ro3 is False:
            message = "Calculating Rule of 5"
            data_df = self._calculate_descriptors(data_df, message, style='ro5')

        elif ro5 is False and ro3 is True:
            message = "Calculating Rule of 3"
            data_df = self._calculate_descriptors(data_df, message, style='ro3')

        elif ro5 is True and ro3 is True:
            print("BOTH ro5 and ro3 is True, RUNNING ro3 only!")

            message = "Calculating Rule of 3"
            data_df = self._calculate_descriptors(data_df, message, style='ro3')

        else:
            pass


        return data_df

    def motif_decorator(self,
                        motif: Union[str, dm.Mol] = None,
                        n_samples_per_trial: int = 10,
                        n_trials: int = 1,
                        sanitize: bool = True,
                        model: str = 'default',
                        ro5: bool = True,
                        ro3: bool = False,
                        np_score: bool = False,
                        do_not_fragment_further: bool = True,
                        seed: int = 42,
                        **kwargs: Optional[Dict[Any, Any]]):
        """

        :param motif: Union[str, dm.Mol]
            Molecule (with attachment points) to decorate.
        :param n_samples_per_trial: int
            The number of new molecules to generate for each randomization.
        :param n_trials: Optional[int]
            Set number of randomization to perform.
        :param sanitize: bool
            Whether to sanitize the generated molecules and check if the scaffold is still present.
        :param model: str
            Set the SAFE model. Set to use the default SAFE model.
        :param ro5: bool
            Score generated molecules to Rule of 5 rules.
        :param ro3: bool
            Score generated molecules to Rule of 3 Lead-Like rules. If both ro5 and ro3 are true, then only ro3 will be
            calculated.
        :param np_score: bool
            Generate a Natural Product score for generated molecule.
        :param do_not_fragment_further: bool
            Whether to fragment the scaffold further or not.
        :param seed: Optional[int]
            Set the random seed to use. This will set the autoregressive model fo SAFE algorithms and set building
            molecules to be deterministic.
        :param kwargs: Optional[Dict[Any, Any]]
        :return:
        """

        """
        There is an issue with using the load_default. To get around it, the function will load the default model 
        internally. This means that a custom model cannot be loaded. This will be troubleshooted and and the functions 
        for custom SAFE models will be wrapped in the future. 
        """
        sf = require("safe", "safe", "SAFEbuild")
        transformers = require("transformers", "safe", "SAFEbuild")

        global safe_model
        if model == 'default':
            safe_model = sf.SAFEDesign.load_default(verbose=True)

        # set safe generation repeatability
        transformers.set_seed(seed)

        generated_smiles = safe_model.motif_extension(motif=motif, n_samples_per_trial=n_samples_per_trial,
                                                      n_trials=n_trials,
                                                      sanitize=sanitize,
                                                      do_not_fragment_further=do_not_fragment_further,
                                                      random_seed=seed,
                                                      **kwargs)

        # output generated smiles and convert into a list of ROMol
        mol_list = [dm.to_mol(x) for x in generated_smiles]

        # calculate sa and np scores for generated_safe molecules
        data_df = self._sa_score_of_generated_safe(generated_smiles, mol_list, np_score=np_score)

        # set options for ro5 or ro3
        if ro5 is True and ro3 is False:
            message = "Calculating Rule of 5"
            data_df = self._calculate_descriptors(data_df, message, style='ro5')

        elif ro5 is False and ro3 is True:
            message = "Calculating Rule of 3"
            data_df = self._calculate_descriptors(data_df, message, style='ro3')

        elif ro5 is True and ro3 is True:
            print("BOTH ro5 and ro3 is True, RUNNING ro3 only!")

            message = "Calculating Rule of 3"
            data_df = self._calculate_descriptors(data_df, message, style='ro3')

        else:
            pass

        # set table to instance variable
        self.data_df = data_df

        return data_df

    def linker_decorator(self,
                         *groups: Union[str, dm.Mol],
                         n_samples_per_trial: int = 10,
                         n_trials: Optional[int] = 1,
                         sanitize: bool = True,
                         model: str = 'default',
                         ro5: bool = True,
                         ro3: bool = False,
                         np_score: bool = False,
                         do_not_fragment_further: Optional[bool] = True,
                         seed: Optional[int] = 42,
                         model_only: Optional[bool] = False,
                         **kwargs: Optional[Dict[Any, Any]]):
        """

        :param groups: Union[str, dm.Mol]
            List of fragments to link together. They are joined in the order provided.
        :param n_samples_per_trial: int
            The number of new molecules to generate for each randomization.
        :param n_trials: Optional[int]
            Set number of randomization to perform.
        :param sanitize: bool
            Whether to sanitize the generated molecules and check if the scaffold is still present.
        :param model: str
            Set the SAFE model. Set to use the default SAFE model.
        :param ro5: bool
            Score generated molecules to Rule of 5 rules.
        :param ro3: bool
            Score generated molecules to Rule of 3 Lead-Like rules. If both ro5 and ro3 are true, then only ro3 will be
            calculated.
        :param np_score: bool
            Generate a Natural Product score for generated molecule.
        :param do_not_fragment_further: Optional[bool]
            Whether to fragment the scaffold further or not.
        :param seed: Optional[int]
            Set the random seed to use. This will set the autoregressive model fo SAFE algorithms and set building 
            molecules to be deterministic.
        :param model_only: Optional[bool]
            Whether ot use the model only ability.
        :param kwargs: Optional[Dict[Any, Any]]
        :return:
        """

        """
        There is an issue with using the load_default. To get around it, the function will load the default model 
        internally. This means that a custom model cannot be loaded. This will be troubleshooted and and the functions 
        for custom SAFE models will be wrapped in the future. 
        """
        sf = require("safe", "safe", "SAFEbuild")
        transformers = require("transformers", "safe", "SAFEbuild")

        global safe_model
        if model == 'default':
            safe_model = sf.SAFEDesign.load_default(verbose=True)

        # set safe generation repeatability
        transformers.set_seed(seed)

        generated_smiles = safe_model.linker_generation(*groups, n_samples_per_trial=n_samples_per_trial,
                                                        n_trials=n_trials, sanitize=sanitize,
                                                        do_not_fragment_further=do_not_fragment_further,
                                                        random_seed=seed, model_only=model_only, **kwargs)
        # output generated smiles and convert into a list of ROMol
        mol_list = [dm.to_mol(x) for x in generated_smiles]

        # calculate sa and np scores for generated_safe molecules
        data_df = self._sa_score_of_generated_safe(generated_smiles, mol_list, np_score=np_score)

        # set options for ro5 or ro3
        if ro5 is True and ro3 is False:
            message = "Calculating Rule of 5"
            data_df = self._calculate_descriptors(data_df, message, style='ro5')

        elif ro5 is False and ro3 is True:
            message = "Calculating Rule of 3"
            data_df = self._calculate_descriptors(data_df, message, style='ro3')

        elif ro5 is True and ro3 is True:
            print("BOTH ro5 and ro3 is True, RUNNING ro3 only!")

            message = "Calculating Rule of 3"
            data_df = self._calculate_descriptors(data_df, message, style='ro3')

        else:
            pass

        # set table to instance variable
        self.data_df = data_df

        return data_df

    def _calculate_descriptors(self, data_df, message, style):
        """
        Script to calculate if generated molecule meets Ro5 or Ro3
        """
        # initialize tqdm for pandas
        tqdm.pandas(desc=message)

        # calculate descriptors and keep only relevant to ro5
        descriptors = dm.descriptors.batch_compute_many_descriptors(data_df['smiles'].apply(dm.to_mol).tolist(),
                                                                    progress=True)
        cols_to_keep = ['mw', 'n_lipinski_hba', 'n_lipinski_hbd', 'clogp', 'qed']
        descriptors = descriptors[cols_to_keep]
        # add descriptors and calculate ro5
        data_df = pd.concat([data_df, descriptors], axis=1)
        data_df[style] = data_df['smiles'].progress_apply(self.ro5)
        return data_df

    @staticmethod
    def _sa_score_of_generated_safe(generated_smiles, mol_list, np_score=False):
        """Support function to clean up results of SAFE generated molecules"""

        # for each ROMol, calculate the SA and NP Score
        sa_score = []
        for mol in mol_list:
            score = Score.sascore(mol)
            score = round(score, 3)  # round score
            sa_score.append(score)
        data = pd.DataFrame({'smiles': generated_smiles, 'SA_score': sa_score})

        # conditional to generate np_score
        if np_score is True:
            np_score = []
            for mol in mol_list:
                npscore = Score.npscore(mol)
                npscore = round(npscore, 3)  # round score
                np_score.append(npscore)
            # add generated smiles, SA score and NP score into a single DataFrame
            data = pd.DataFrame({'smiles': generated_smiles, 'SA_score': sa_score, 'NP_score': np_score})

        return data


class BRICSBuild(Score):
    def __init__(self):
        super().__init__()  # call SAScore's __init__
        print("NOT WRITTEN YET!")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
