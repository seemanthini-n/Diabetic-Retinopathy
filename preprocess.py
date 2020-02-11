import os

os.chdir(r'C:\Users\Moorthy\Documents\pywork\retina')

import pandas as pd

import shutil

df=pd.read_csv('trainLabels.csv')

frm=r'C:\Users\Moorthy\Documents\pywork\retina\train'
dst=r'C:\Users\Moorthy\Documents\pywork\retina\trnsep'

l1=os.listdir(dst)

l2=[x.replace('.jpeg','',1) for x in l1]

df2=pd.DataFrame(l2)
df2.columns=['image']

df3=df2.merge(df)

df3.level.value_counts()

for i in range(len(df3)):
    frmnew=frm+"\\"+df3.iloc[i,0]+".jpeg"
    x=df3.iloc[i,1]
    if x==0:
        add2="No_DR"
    elif x==1:
        add2="Mild"
    elif x==2:
        add2="Moderate"
    elif x==3:
        add2="Severe"
    elif x==4:
        add2="Proliferative_DR"
   
    dstnew=dst+"\\"+add2
    
    shutil.copy2(frmnew, dstnew)


