# refitt_v1

## Running live
Simply execute run.py

Results will be placed in a subfolder ZTF/ZTF_%Y-%m-%dT%H%M
Priority lists are in a deeper subfolder /priorty

Note: prioritize_old.py is legacy and will be removed in the future

###Updates:
- predicting class and forecasting on class
- loss is error on full forecast
- Cross corr align

###Future improvements:
- Forecasts for bright transients and decaying transients is not good
- System uses LC only - host information would help (redshift+env properties)
- Need to have more expansive and uniform training library

##Training
###Reference Library
files in <class_name>/train/ has json files for reference LCs
record to 60 days from trigger in g and r

these are simulations made with wrangle_data_for_library.py and make_library_ZTF.py
but forgot to record details on how to go about making these

###Setup
1. Build CAE reps for each library LC from horizon-window days by running make_reps.py.
takes 5-6 hours per epoch (5000 sims per class) on 1 full brown node

2. Make ball trees by running make_balltrees.py
takes ~2 hours per epoch and can be run in parallel

3. Find k for each integer time since trigger
run find_kNN.py via submit_find_kNN.py 
takes 4 days per epoch but can be run in parallel

4. Run kNN_optimal.py to find best k from obj_<phase>.npy files written in step 3.
Useful to examine loss in obj.png
Loss curves are not converged for less than 9 days since trigger
Overall results are only reliable after 10 days since trigger – hardsetting to 10’s k for earlier times

###Laundry list:
Use spec class or probabilistic class
Use host info (redshift, offset, morphology)

