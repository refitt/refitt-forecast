import pandas as pd
from sklearn.metrics import confusion_matrix
import refitt

tst_list=['7','40']
for tst in tst_list:
  classf=pd.read_pickle('classf_'+tst+'.pkl')
  for index, row in classf.iterrows():
       if row[0] in row[1][0]:
           row[1]=row[0]
       else:
           row[1]=row[1][0][0] #first class
  
  cs=list(refitt.lib_classes.keys())
  C=confusion_matrix(classf[0],classf[1],normalize='true',labels=cs)
  
  import seaborn as sns
  import matplotlib.pyplot as plt
  plt.figure()
  sns.heatmap(C,annot=True,xticklabels=cs,yticklabels=cs)
  plt.savefig('conf_'+tst+'.png')

