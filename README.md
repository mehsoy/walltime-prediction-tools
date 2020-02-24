# walltime-prediction-tools

This Repo contains tools and code for predictiong HPC Job wall times.
Check LICENSE file for licensing.

Authors:  
   Mehmet Soysal(mehmet.soysal@kit.edu)  
   Markus Götz(markus.goetz@kit.edu)
#

dirs  
```
 helpers/                               contains small helper functions 
 results-collection/                    a collection of results  
 results-collection/tpot-collection/    collection of saved model from tpot runs  
 workloads/                             place swf workloads here, contains ATM only fh1 workloads  
                                        for other workloads execute "download-workloads.sh"  
 results/                               just empty dir for writing the results  
``` 

Usage:

either install packages from requirements.txt in a venv or into system/user  
dependacies should be installed automatically by pip/conda etc...

$ pip3 install -r requirements.txt  
Important:   
Probably auto-sklearn==0.6.0 is not available with pip 

#in venv
git clone  https://github.com/automl/auto-sklearn  
cd auto-sklearn/  
python3 setup.py install  

 
available predictors:   
+ walltime-predictioner-alea.py: reimplement the ALEA builtin predictor (max of last 5 ratios = T_run/T_Req)   
+ walltime-predictioner-last5avg.py: similar to alea but using avg of last 5 ratios  
+ walltime-predictioner-keras-hyperas.py: Example for using tensorflow+Keras+hyperas  
+ walltime-predictioner-keras-mg.py: Contribution from Markus Götz  
+ walltime-predictioner-keras.py: Example with tensorflow+keras  
+ walltime-predictioner-scikit.py: using different Regressor form scikit package (see Code)  
+ walltime-predictioner-scikit-autosklearn.py: using auto-sklearn as AUTOML package  (https://automl.github.io/auto-sklearn/master/)  
+ walltime-predictioner-scikit-tpot.py: using TPOT as AUTOML package https://epistasislab.github.io/tpot/  
+ walltime-predictioner-scikit-graphviz.py: small snippets to create png with graphviz package etc ...  




