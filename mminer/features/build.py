"""
Scripts to generate molecules de novo
"""
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to reduce error messages during SAFE molecule generation

from typing import Optional, Union, Dict, Any
from rdkit.Chem import BRICS
import pandas as pd
import safe as sf
import datamol as dm
from tqdm import tqdm
import transformers

from mminer.utils.rdkit_contrib.SA_Scorer import sascorer
from mminer.utils.rdkit_contrib.NP_Scorer import npscorer

__all__ = ["SAFEbuild", "BRICSBuild", 'Score']


class Score:
    """
    Calculate SA Score https://greglandrum.github.io/rdkit-blog/posts/2023-12-01-using_sascore_and_npscore.html
    """

    def __init__(self, data: Union[pd.DataFrame, dm.Mol] = None):
        if isinstance(data, pd.DataFrame):
            self.data_df = data
        elif isinstance(data, dm.Mol):
            self.data = data
        else:
            self.data_df = None
            self.data = None

    @staticmethod
    def sascore(mol: Union[str, dm.Mol] = None):
        """
        Calculate Synthetic Accessibility Score of a molecule.
        :param mol: Union[str, dm.Mol]
            Molecule smiles string or rdkit.Chem.Mol object.
        :return:
        """
        # check if mol is string. If so, convert to ROMol.
        if isinstance(mol, str):
            mol = dm.to_mol(mol)

        sa_score = sascorer.calculateScore(mol)
        return sa_score

    @staticmethod
    def npscore(mol: Union[str, dm.Mol] = None):
        """
        Calculate Natural Product-likeness Score of a molecule.
        :param mol: Union[str, dm.Mol]
            Molecule smiles string or rdkit.Chem.Mol object.
        :return:
        """
        # check if mol is string. If so, convert to ROMol.
        if isinstance(mol, str):
            mol = dm.to_mol(mol)

        fscore = npscorer.readNPModel()
        np_score = npscorer.scoreMol(mol, fscore)
        return np_score

    def ro5(self, mol: Union[str, dm.Mol] = None):
        """
        Calculate Rule of 5 Compliance of a molecule.
        :param mol: Union[str, dm.Mol]
            Molecule smiles string or rdkit.Chem.Mol object.
        :return:
        """
        # check if mol is string. If so, convert to ROMol.
        if isinstance(mol, str):
            mol = dm.to_mol(mol)

        # calculate molecule description
        descriptors = dm.descriptors.compute_many_descriptors(mol)

        # assess ro5 values
        mw = descriptors.get('mw')
        hbd = descriptors.get('n_lipinski_hbd')
        hba = descriptors.get('n_lipinski_hba')
        clogp = descriptors.get('clogp')

        # check ro5 values
        if hbd and clogp <= 5 and hba <= 10 and mw <= 500:
            return 'True'
        else:
            return 'False'

    def ro3(self, mol: Union[str, dm.Mol] = None):
        """
        Calculate Rule of 3 Compliance of a molecule for Lead-Like molecules.
        :param mol: Union[str, dm.Mol]
            Molecule smiles string or rdkit.Chem.Mol object.
        :return:
        """
        # check if mol is string. If so, convert to ROMol.
        if isinstance(mol, str):
            mol = dm.to_mol(mol)

        # calculate molecule description
        descriptors = dm.descriptors.compute_many_descriptors(mol)

        # assess ro5 values
        mw = descriptors.get('mw')
        hbd = descriptors.get('n_lipinski_hbd')
        hba = descriptors.get('n_lipinski_hba')
        clogp = descriptors.get('clogp')
        rotatable_bonds = descriptors.get('n_rotatable_bonds')
        qed = descriptors.get('qed')

        self.descriptor_dict = {
            'mw': mw,
            'n_lipinski_hba': hba,
            'n_lipinski_hbd': hbd,
            'clogp': clogp,
            'n_rotatable_bonds': rotatable_bonds,
            'qed': qed
        }

        # check ro3 values
        if hbd and hba and clogp and rotatable_bonds <= 3 and mw <= 300:
            return 'True'
        else:
            return 'False'

    # todo update with other score options
    def prep_legend(self, data: pd.DataFrame = None, np_score: bool = False, qed: bool = False, clogp: bool = False,
                    scores: bool = False):
        """
        Modify the pd.DataFrame of generated smiles. This will output a list to be used as a figure legend when drawing
        molecules from the generated smiles (dm.to_imate()).
        :param data: pd.DataFrame
            pd.DataFrame of generated smiles. To ensure matching, run method AFTER processing the data.
        :param np_score: bool
            To determine to include the np_score score.
        :param qed:
        :param clogp:
        :param scores: bool
            To print all scores as figure legend.
        :return:
        """
        if data is None:
            data = self.data_df

        # convert float to string
        data['SA_score'] = data['SA_score'].astype(str)
        # convert score to list
        sa_score = data['SA_score'].tolist()
        # add prefix to each item in list
        sa_score = ['SA Score: ' + score for score in sa_score]

        # option for np_score
        if np_score is True:
            data['NP_score'] = data['NP_score'].astype(str)
            np_score = data['NP_score'].tolist()
            np_score = ['NP Score: ' + score for score in np_score]

            # Combine the list into a single list for the legend
            legends = [f"{item1}\n\n\n{item2}" for item1, item2 in zip(sa_score, np_score)]

            return legends

        # if np_score is false
        return sa_score


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
