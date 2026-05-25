"""
The following script will be used to canonize molecules. This should fix any potential kekulization errors.
"""
from urllib.parse import quote
import requests
from tqdm import tqdm

__all__ = ['sanitizer']


def sanitizer(df, smi_col=None):
    """
    Sanitize molecules using PubChem PUG-REST API. Query is a Dataframe with a smiles column. The standardized smiles
    will be concacted to the dataframe upon output

    Ref:
    https://pubchem.ncbi.nlm.nih.gov/standardize/standardize.cgi
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0293-8

    :param df: Pandas.DataFrame
        Pandas DataFrame containing molecules for processing.
    :param smi_col: String
        Column name containing SMILES string.
    :return:
    """

    results = []
    smi_list = df[smi_col].tolist()
    for molecule in tqdm(smi_list, desc="Standardizing Molecule"):
        try:
            mol = molecule.replace(
                "/", "."
            )  # some smiles string uses '/' which is incompatible for URLs
            encoded_smi = quote(mol)  # URL encode the SMILES string

            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/standardize/smiles/{encoded_smi}/json"

            # Send a GET request to the PUG URL
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad responses

            # Parse JSON response
            json_data = response.json()

            # Extract relevant information from the JSON data
            standardized_smi = json_data["PC_Compounds"][0]["props"][2]["value"]["sval"]

            results.append(standardized_smi)

        except requests.exceptions.RequestException as e:
            print(f"Error making request to PUG: {e}")
            results.append("None")

    return results


if __name__ == "__main__":
    import doctest

    doctest.testmod()
