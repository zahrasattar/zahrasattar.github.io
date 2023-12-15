#!/usr/bin/env python
# coding: utf-8

# ## Week 1: Getting started with Anaconda, Jupyter Notebook and Python 

# a) Why i chose to join this course? - i chose to join this course because i find AI quite interesting and as it is becoming more prevalent in society i wanted to learn more about it. 
# 
# b) I have no prior experience with AI or with coding such as Python or any coding similar. I do have some basic knowledge and skills with HTML coding. 
# 
# c) What i expect to learn from the course:
# - More about how AI operates and functions 
# - Some ethics regarding the uses of AI
# - New ventures and emerging uses of AI

# In[1]:


"Hello, World!"


# In[10]:


message = "Hello World!"

print (message[0])


# What happens if you print messasge + message ? - it displays hello world twice 
# 
# What happens when you print message*3 ? - it prints hello world three times 
# 
# What happens when you print message [0] ? - it just prints H
# 
# How to change 'message' as a variable name? - i think message is a good variable name, however if it is changed to a new variable name, then that same name will need to be used in the print line as well otherwise the old 'message' will print instead. 

# In[2]:


from IPython.display import * 


# In[3]:


YouTubeVideo("dQw4w9WgXcQ")


# In[19]:


import webbrowser #this is importing a library
import requests  #this is importing a library

print("Shall we hunt down an old website?") #this is printing the text
site = input("Type a website URL: ") #this is assigning a value to a variable 
era = input("Type year, month, and date, e.g., 20150613: ") #this is assigning a value to a variable 
url = "http://archive.org/wayback/available?url=%s&timestamp=%s" % (site, era) #this is importing a library?
response = requests.get(url)
data = response.json()
try:
    old_site = data["archived_snapshots"]["closest"]["url"]
    print("Found this copy: ", old_site) #this is printing the text  
    print("It should appear in your browser.") #this is printing the text
    webbrowser.open(old_site)
except:
    print("Sorry, could not find the site.") #this is printing the text

