#!/usr/bin/env python
# coding: utf-8

# ## Week Two: Exploring Data in Multiple Ways

# In[1]:


from IPython.display import Image


# In[2]:


Image ("picture1.jpg")


# In[3]:


from IPython.display import Audio


# In[4]:


Audio ("audio1.mid") 


# In[5]:


Audio ("audio2.ogg") 


# In[6]:


#The sound file audio2.ogg is owned by Artoffuge Mehmet Okonsar. Remember to add the attribution and include a comment in the cell (in a cell of type code, you do this by prefixing text with a hash tag) to flag the following licensing information about audio2.ogg:
#This file is licensed under the Creative Commons Attribution-Share Alike 3.0 Unported license.
#You are free: 
#•	to share – to copy, distribute and transmit the work
#•	to remix – to adapt the work
#Under the following conditions: 
#•	attribution – You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
#•	share alike – If you remix, transform, or build upon the material, you must distribute your contributions under the same or compatible license as the original.
#The original ogg file was found at the url: 
#https://en.wikipedia.org/wiki/File:GoldbergVariations_MehmetOkonsar-1of3_Var1to10.ogg


# ### Matplotlib 

# In[7]:


from matplotlib import pyplot
test_picture = pyplot.imread("picture1.jpg")
print("Numpy array of the image is: ", test_picture)
pyplot.imshow("test_picture_filtered")


# In[ ]:




