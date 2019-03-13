#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
from simpledbf import Dbf5
import numpy as np


# In[2]:


#Download the INSEE Database of all the name given between 1900 and 2017 : for each (name,years), it shows the number of occurence by gender
table=Dbf5('nat2017.dbf', codec='cp1252')
dfNameAndGender = table.to_dataframe()
dfNameAndGender.columns


# In[3]:


dfNameAndGender["annais"]=pd.to_numeric(dfNameAndGender["annais"],errors="coerce")


# In[4]:


#We filter the database of name to only have names of people in the active population : between 1950 and 2000
filter= (dfNameAndGender["annais"]>=1950) & (dfNameAndGender["annais"]<=2000)
df = dfNameAndGender.where(filter).groupby(['preusuel','sexe'],as_index=False)['nombre'].sum()


# In[5]:


#As the INSEE database doesn't give us if a name is female or male, we compute a proportion on all of years of studies
df["total"]= df.groupby(["preusuel"],as_index=False)["nombre"].transform("sum")
df["proportion"]=df["nombre"]/df["total"]
#We then decide to associate a gender to all the name that are give mostly (>=70%) to one of the gender. The other are said to be undetermined (for exemple Dominique)
filter1 = (df["proportion"]>=0.70)
df1=df.where(filter1)

list_prenom = df1["preusuel"].unique()

df2=df[~df["preusuel"].isin(list_prenom)]
df3= df2.groupby(["preusuel"],as_index=False)["sexe"].max()
df3["sexe"]=df3["sexe"].apply(lambda x :0)
df1=df1[["preusuel","sexe"]]

#We produce a final database with all the "determined" name
dfFinal=pd.concat([df1,df3])


# In[82]:


'''
    Input : string
    Output : Number, 0:Undertermined, 1:Male, 2:Female
    From the first word of a CV, this function determined if it's Male, Female or Undetermined. To do so, it looks for each word in our name database.
    If it doesn't find a gender for any of the word, or if it finds more than one gender, it returns undetermined. Else it returns the gender it found.
    We stop looking when we compute a word that signify an adress, as more often than not the name is situated before the adress. It also helps with the accuracy,
    as some street name are people name, which leads to more undetermined case.
'''

def gender_cvs(cv_first_words):
    list_gender=[]
    words=cv_first_words.upper().split()
    i=0
    while i<len(words) and words[i] not in ["RUE","AVENUE","IMPASSE","BOULEVARD","ROUTE"]:
        if(not dfFinal[dfFinal["preusuel"]==words[i]].empty):
            list_gender.append(int(dfFinal[dfFinal["preusuel"]==words[i]]["sexe"].values[0]))
        i+=1
    if len(list_gender)==0:
        return 0
    if list_gender.count(1)>=1:
        if list_gender.count(2)>=1:
              return 0
        else:
              return 1
    else:
        return 2


# In[112]:


#We charge our data CSV and transform it to have only to column : a column with the Id and a column with the content, parsed from pdf
csv_data = pd.read_csv("Data/RECBUD/RDS/V_CentraleSup_CV_REC001_tab.csv",sep='\t')
csv_data=csv_data[["MATRICULEINT","TXT_CONTENU1","TXT_CONTENU2","TXT_CONTENU3","TXT_CONTENU4","TXT_CONTENU5"]]


# In[115]:


cropped_data=csv_data[:].copy().fillna(" ")
cropped_data["TXT"]=cropped_data["TXT_CONTENU1"]+cropped_data["TXT_CONTENU2"]+cropped_data["TXT_CONTENU3"]+cropped_data["TXT_CONTENU4"]+cropped_data["TXT_CONTENU5"]
cropped_data=cropped_data[["MATRICULEINT","TXT"]]
cropped_data.dropna(inplace=True)




# In[116]:


#We create a new column Gender by applying our function to the text
cropped_data.loc[:,"GENRE"]=cropped_data["TXT"].apply(lambda x : gender_cvs(x[:20]))


# In[117]:





# In[ ]:





# In[ ]:





# In[118]:


#We create a dataframe with all the male CV
filter_male=(cropped_data["GENRE"]==1)
df_male=cropped_data[filter_male]
#df_male


# In[119]:


#We create a dataframe with all the female CV
filter_female=(cropped_data["GENRE"]==2)
df_female=cropped_data[filter_female]
#df_female


# In[120]:


#We create then export to csv a dataframe with the Id, the content and the gender of the CV by concatenating and shuffling the female and male dataframe
dfData=pd.concat([df_male,df_female]).sample(frac=1)
dfData.to_csv("data_with_gender_rec.csv",index=False)


# In[34]:


filter_undetermined=(cropped_data["GENRE"]==0)
df_undetermined=cropped_data.where(filter_undetermined)
#df_undetermined

