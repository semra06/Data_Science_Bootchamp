#!/usr/bin/env python
# coding: utf-8

# ## Görev 2:
# 

# In[89]:


text = "The goal is to run data into information, and information into insight"

#for i in range(len(text)):
#   new_text += text[i].upper()
#    if text[i] == " ":
#        new_text += ","
#print(new_text)
new_string = "".join(["," if text[i] == " " else text[i].upper() for i in range(len(text))])
new_string


# ## Görev 3: 

# In[35]:


lst = ["D","A","T","A","S","C","I","E","N","C","E"]
len(lst)


# In[39]:


lst = ["D","A","T","A","S","C","I","E","N","C","E"]
lst[0]
lst[10]


# In[51]:


lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]
new_list = []

[new_list.append(lst[i]) for i in range(0, 4)]

print(new_list)


# In[54]:


lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]
lst[0:4]


# ## Görev 4: 

# In[56]:


dict = {"Cristian": ["American", 18],
        "Daisy": ["England", 12],
        "Antonio": ["Spain", 22],
        "Dante": ["Italy", 25]}
dict.keys()


# In[58]:


dict = {"Cristian": ["American", 18],
        "Daisy": ["England", 12],
        "Antonio": ["Spain", 22],
        "Dante": ["Italy", 25]}
dict.values()


# In[62]:


dict = {"Cristian": ["American", 18],
        "Daisy": ["England", 12],
        "Antonio": ["Spain", 22],
        "Dante": ["Italy", 25]}
dict["Daisy"] = ["England", 13]
dict["Ahmet"] = ["Turkey", 24]
dict.pop("Antonio")
dict


# ## Görev 5:

# In[69]:


l = [2, 13, 18, 93, 22]
def func(l):
    even_list = []
    odd_list = []
    for i in range(len(l)):
        if i %2 == 0:
            even_list.append(l[i])
        else:
            odd_list.append(l[i])
    return even_list, odd_list
func(l)            
    


# ## Görev 6:

# In[70]:


ogrenciler = ["Ali", "Verli", "Ayşe", "Talat", "Zeynep", "Ece"]
for ogrenci in enumerate(ogrenciler):
    print(ogrenci)


# In[85]:


ogrenciler = ["Ali", "Verli", "Ayşe", "Talat", "Zeynep", "Ece"]
for index, ogrenci in enumerate(ogrenciler):
    if index < 3:
        print(f"Mühendislik Fakültesi {index+1} . öğrenci:", ogrenci)
    else:
        print(f"Tıp Fakültesi {index+1} . öğrenci:", ogrenci)


# ## Görev 7:

# In[87]:


ders_kodu = ["CMP1005", "PYS1001", "HUK1005", "SEN2204"]
kredi = [3,4,2,4]
kontenjan = [30,75,150,25]
for ders, krediler, kontenjanlar in zip(ders_kodu, kredi, kontenjan):
    print(f"Kredisi {krediler} olan {ders} kodlu dersin kontenjanı {kontenjanlar} kişidir.")


# ## Görev 8:

# In[95]:


kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul" ])
if kume2.issuperset(kume1):
    print(kume2 - kume1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




