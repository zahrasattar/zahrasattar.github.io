#!/usr/bin/env python
# coding: utf-8

# # Generating Text with Neural Networks
# 

# This code uses neural networks to create text inspired by Shakespeare's works. It uses Tensorflow, a Python library to build these neural networks. This model shows ways in which humanities are involved in Large Language Models (LLM) beyond general generative AI. Projects regarding specialisied datasets for machine learning requires having effective resources as it can be quite difficult to build. In this project, i reduced the training data because my personal computer does not have the capacity that is required for this project to be done successfully, after many crashes i decided reducing the training data would be the better option. This however, does reflect and show with many outputs as some appear being quite confusing and rather nonsensical. This shows that it is hard to do more projects like this in the humanities with few and limited resources. Neural networks are complex however, i added comments throughout to simplify it for myself and others. Overall, this project highlights the need for humanities students to learn digital skills for modern technology. 

# # Getting the Data

# In[2]:


import tensorflow as tf

shakespeare_url = "https://homl.info/shakespeare"  # shortcut URL
filepath = tf.keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()


# In[3]:


print(shakespeare_text[:80]) # not relevant to machine learning but relevant to exploring the data


# # Preparing the Data

# The code allows for computational analysis as it take Shakespeare’s dataset and coverts it into a computational format, helping with things like recognising patterns and sentiments. It sets up three parts: training to teach the computer, validation to check what it learns, and a test to see how well it uses the learning. The goal is to train a computer to write like Shakespeare. The data preparation, though time-consuming, is crucial as the computer learns from it, stressing the importance of understanding what we give to the model. In this snippet of code the sequence length is set to 100, and a random seed is initialised for reproducibility using TensorFlow. The training set, derived from the initial 125,000 elements of the encoded data, is shuffled for randomness and serves to teach the program. The validation set (125,000 to 132,500) assesses the program's learning, while the test set (from 132,500 onward) evaluates its application of learned knowledge. These groups are intended to train and test a computer program on transformed Shakespearean text, helping it learn to write in Shakespeare's style. Though getting the data ready takes time, it's crucial because the computer learns from it. It underscores how important it is to know what information we're giving to the program.

# In[4]:


text_vec_layer = tf.keras.layers.TextVectorization(split="character",
                                                   standardize="lower")
text_vec_layer.adapt([shakespeare_text])
encoded = text_vec_layer([shakespeare_text])[0]


# In[5]:


print(text_vec_layer([shakespeare_text]))


# In[6]:


encoded -= 2  # drop tokens 0 (pad) and 1 (unknown), which we will not use
n_tokens = text_vec_layer.vocabulary_size() - 2  # number of distinct chars = 39
dataset_size = len(encoded)  # total number of chars = 1,115,394


# In[7]:


print(n_tokens, dataset_size)


# In[8]:


def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
    if shuffle:
        ds = ds.shuffle(100_000, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)


# In[9]:


length = 100
tf.random.set_seed(42)
train_set = to_dataset(encoded[:125_000:], length=length, shuffle=True,
                       seed=42)
valid_set = to_dataset(encoded[125_000:132_500], length=length)
test_set = to_dataset(encoded[132_500:], length=length)


# # Building and Training the Model

# In this part, the aim is for the model to learn through using organized Shakespearean text from the previous section. The code ensures reproducibility by setting a random seed. The code uses TensorFlow’s Keras API to build and train the neural network for language modelling. The program is structured with layers: transforming characters into numerical values, identifying patterns using GRU, and predicting the next character. The training runs for ten epochs, using specified training and validation datasets. Adjusting epochs is a common step, but it depends on the dataset and model complexity. More epochs and data optimize results but are more demanding, requiring more computational power and resources. The decision depends on desired accuracy. For Shakespeare enthusiasts, increasing data and epochs may be preferred, but I reduced data for simplicity and for better efficiency as my computer does not have the computational resources required to produce effective outputs in an efficient timescale. The goal is to create a model mimicking Shakespeare's writing. This process is time-consuming, as it waits for the computer to analyse the data based on our instructions. We provide the data and simply wait for it to load.
# 

# In[10]:


tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Dense(n_tokens, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    "my_shakespeare_model", monitor="val_accuracy", save_best_only=True)
history = model.fit(train_set, validation_data=valid_set, epochs=10,
                    callbacks=[model_ckpt])


# In[11]:


shakespeare_model = tf.keras.Sequential([
    text_vec_layer,
    tf.keras.layers.Lambda(lambda X: X - 2),  # no <PAD> or <UNK> tokens
    model
])


# # Generating Text

# The code generates text based from the inputs in the previous section by using a trained language model, shakespeare_model. The text "To be or not to b" is inputed and allows for the model to predict probabilities for following outputs. This part allows us to test whether the text aligns with Shakespeare. 

# In[12]:


y_proba = shakespeare_model.predict(["To be or not to b"])[0, -1]
y_pred = tf.argmax(y_proba)  # choose the most probable character ID
text_vec_layer.get_vocabulary()[y_pred + 2]


# In[13]:


log_probas = tf.math.log([[0.5, 0.4, 0.1]])  # probas = 50%, 40%, and 10%
tf.random.set_seed(42)
tf.random.categorical(log_probas, num_samples=8)  # draw 8 samples


# In[14]:


def next_char(text, temperature=1):
    y_proba = shakespeare_model.predict([text])[0, -1:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    return text_vec_layer.get_vocabulary()[char_id + 2]


# In[15]:


def extend_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text


# In[16]:


tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU


# In[17]:


print(extend_text("To be or not to be", temperature=0.01))


# In[18]:


print(extend_text("To be or not to be", temperature=1))


# In[19]:


print(extend_text("To be or not to be", temperature=100))


# As shown above the statement is very nonesensical and evidently unable to be read. It is not true to Shakespeare's work. 

# The model's performance is notable lacking in quality, and a big reason for that will be due to my decision to use less data for faster loading. When I compared my results with those of my peers, their data was clearer and showed how utlising computational resources in which this model demands, produces much more qualititive outcomes. This difference raises concerns about how useful the model is in humanities. 
