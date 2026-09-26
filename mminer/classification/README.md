# PROTAC Classification

This folder keeps python files to run the classification model. 

Currnetly the model is set to classify small molecules. This will be segmented into 3 sections:
- Warhead classification
- E3 ligase classification
- Linker classification

Models will be contacted to "generate" the most likely PROTAC for a given target.

Compounds are first processed using the 'process_dataset.py' file. 

The models will be run using the sklearn-models first.

After testing with sklearn models, additional models will be incorporated - Keras, XGBoost, and pytorch