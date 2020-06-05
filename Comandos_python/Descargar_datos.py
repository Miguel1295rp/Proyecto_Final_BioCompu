import requests
import json
from pandas.core.frame import DataFrame
pheno_03_query = {'mesh_id':'D006262'}  ## -- to get statistics on MeSH ID D006262
url = 'https://gmrepo.humangut.info/api/getAssociatedSpeciesByMeshID'
pheno_03 = requests.post(url, data=json.dumps(pheno_03_query))
pheno_03_cont = pheno_03.json()

## --get DataFrame
phenotyp_assoc_species = DataFrame(pheno_03.json())

## --show data header of the resulting DataFrame
list(phenotyp_assoc_species)
#print(phenotyp_assoc_species)
print
phenotyp_assoc_species
phenotyp_assoc_species.to_csv('filename.csv')
files.download('filename.csv')