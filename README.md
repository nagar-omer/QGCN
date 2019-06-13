# QGCN
machine learning on graph based datasets,

this package provide tools for deep-learning on graph based datasets, it can be used for regression as well as for multi-class classification. 
required packages:
- pytorch (optional use of GPU)
- numpy

### How to use 
to use this package you will need to provide the following files as input
- graphs csv file - files that contains the graphs for input and their labels
  the format of the file is flexible, but it must contain headers for any columns, and the there must be a column provided for:
  - graph id
  - source node id
  - destination node id
  - label id (every graph id can be attached to only one label)
- external data file - external data for every node 
    the format of this file is also flexible, but it must contain headers for any columns, and the there must be a column provided for:
    **note!! every node must get a value**
    - graph id
    - node id
    - column for every external feeture (if the value is not numeric then it can be handeled with embeddings)
    
example for such files:
graph csv file:
g_id,src,dst,label
6678,_1,_2,i
6678,_1,_3,i
6678,_2,_4,i
6678,_3,_5,i

external data file:
g_id,node,charge,chem,symbol,x,y
6678,_1,0,1,C,4.5981,-0.25
6678,_2,0,1,C,5.4641,0.25
6678,_3,0,1,C,3.7321,0.25
6678,_4,0,1,C,6.3301,-0.25

once this is ready, all left to do is to create parameters for the model and run it:
examples for param_files are aveilable under params directory

example for running the code 
```python
# create dataset
ext_train = ExternalData(AidsAllExternalDataParams())
data_params = AidsDatasetAllParams()
aids_train_ds = BilinearDataset(data_params, external_data=ext_train)

# create binary-model
binary_model = LayeredBilinearModule(AidsLayeredBilinearModuleParams(
    ftr_len=aids_train_ds.len_features, embed_vocab_dim=ext_train.len_embed()))
    
# create trainer
activator = BilinearActivator(binary_model, AidsBilinearActivatorParams(), aids_train_ds)
activator.train()
```
