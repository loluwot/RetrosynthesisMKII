Using neural networks to predict reasonable retrosynthetic steps. Implementation of this paper: [https://doi.org/10.1002/chem.201605499](https://doi.org/10.1002/chem.201605499)

Next steps:
Implement [https://doi.org/10.1038/nature25978](https://doi.org/10.1038/nature25978) to utilize retrosynthetic step prediction to create a fully functional retrosynthetic pathway designer.

To use:
Download open-source database of chemical reactions extracted from US patents here: [https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873](https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873)

Extract 1976_Sep2016_USPTOgrants_smiles.7z into a folder named DATA.

Run data.py.

Run multi.py.

Run main.py.

Run RESULTS/train.py.

The model should now be stored in RESULTS.
