# REFITT
**Recommender Engine For Intelligent Transient Tracking**

## Setup
### Packages/Dependencies
- george
- Keras
- sncosmo, sfdmap
- elasticsearch_dsl
- antares-client
- tslearn (optional)

### Prep
pip install .

tar xzf data/lib_gen/train_lcs.tar.gz -C data/lib_gen

### Generate Library
Run make_library_ZTF.py in place using submit_make_library_ZTF.py.
This will generate training LCs under data/class. Takes 2.5hrs for Ia and <1hr for rest

### Training
1. Build CAE reps for each library LC by running make_reps.py.
takes ~11-15 hours per epoch on 1 full brown node

2. Make ball trees by running make_balltrees.py.
takes ~5-6 hours per epoch (per core) and can be run in parallel

3. Find k for each integer time since trigger by running find_kNN.py via submit_find_kNN.py.

4. Run kNN_optimal.py to find best k from obj_phase.npy files written in step 3.
Useful to examine loss in obj.png.

## Running live
Simply run run.py

Results will be placed in a subfolder ZTF/ZTF_%Y-%m-%dT%H%M.
Priority lists are in a deeper subfolder /priority

Note: prioritize_old.py is legacy and will be removed in the future

# Notes:
## Disclaimer:
Many scripts are written for slurm and will need a bit of work to get to run on PCs

## Data files for above
One only needs the training LCs, balltree_AE_*, and kNN_optimal.json files to run.
Some of these files are large and therefore not included but can be made available on request.
This means you can skip generating the library and training (above).

### Major changes from Sravan+2020:
- GP fit before putting through Xception
- predicting class and forecasting on class
- Cross corr align
- loss is L1 error on full forecast

### Laundry list:
- Forecasts for bright transients is not good
- Use host info (redshift, offset, morphology)

