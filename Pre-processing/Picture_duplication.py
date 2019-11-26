#!/usr/bin/env python
# coding: utf-8

# # Duplication des photos

# In[96]:


import json
import sys
import os
import coco
import download
from cache import cache
from PIL import Image

from IPython.display import clear_output
# example of random rotation image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')

import os.path
from os import path


# # Chargement des données maison

# In[76]:


# Dossier de téléchargement du dataset Microsoft Coco
coco.set_data_dir("data/coco/")


# In[77]:


# Récupération des noms de fichiers (filenames) et des descriptions (captions) 
# du jeu d'entraînement (training set) à l'aide du module coco.py
# returns ids, filename, caption
# si pb ouvre dans notepad++ encoding UTF8, save as
_, filenames_train, captions_train = coco._load_records(train=True)


# In[78]:


# Vérification des structures de données # 875
num_images_train = len(filenames_train)
num_images_train


# In[79]:


filenames_train[1]


# In[80]:


len(captions_train)


# In[81]:


# group of all labels for each images
captions_train[273]


# In[82]:


# On obtient une liste de listes
num_captions_train = 0
min_caption_number = 10
max_caption_number = 0
for caption_list in captions_train:
    num_captions_train += len(caption_list)
    if len(caption_list) < min_caption_number:
        min_caption_number = len(caption_list)
    if len(caption_list) > max_caption_number:
        max_caption_number = len(caption_list)
print("Total caption number: " + str(num_captions_train))
print("Min caption number: " + str(min_caption_number))
print("Max caption number: " + str(max_caption_number))


# In[83]:


# Quantité moyenne de captions par image
mean_caption_number = num_captions_train / num_images_train
mean_caption_number


# ## Data augmentation : flip left to right the train dataset

# ### Duplicate images

# In[112]:


# Load and resize image

def load_flip_train_image(dir_in, dir_out, filenames_train, num_images_train, num_aug=9):
    """
    Chargement des images à partir d'un dossier,
    Copie des images dans un nouveau dossier, 
    Retournement et duplication des images dans ce même nouveau dossier
    """
    
    # Vérifie que la duplication n'a pas été déjà faite
    if  False:
        print("Number of images already changed")
        
    else :     
        for i in range(len(filenames_train)):
            
            filename = filenames_train[i]

            print(filename)
            if(os.path.exists(str(dir_out+filename))):
                continue
            
            path = os.path.join(dir_in, filename)
    
            # Load one image w/ PIL.
            img = Image.open(path)
            
            # convert to numpy array
            data = img_to_array(img)
            # expand dimension to one sample
            samples = expand_dims(data, 0)
            # create image data augmentation generator
            datagen = ImageDataGenerator(rotation_range=90)
            # prepare iterator
            it = datagen.flow(samples, batch_size=1)
            # generate samples and plot
            for i in range(num_aug):
                batch = it.next()
                # convert to unsigned integers for viewing
                image = batch[0].astype('uint8')
                
                image_out = Image.fromarray(image, 'RGB')
                current_index = str(i)
                image_out.save(dir_out +"t"+current_index+"_"+filename)
            
            img.save(dir_out + filename)

        print (len(os.listdir(dir_out)))


# In[113]:


# take several minutes to run
dir_dupli = ("data/coco_new/train2017/")
dir_ori = ("data/coco/train2017/")
load_flip_train_image(dir_in = dir_ori, dir_out = dir_dupli, 
                      filenames_train = filenames_train, num_images_train = num_images_train)


# In[115]:


len(os.listdir("data/coco_new/train2017/"))


# In[116]:


# duplicate annotations file
def annotations_duplication(annot_dir_in, annot_dir_out, filename_captions, filename_output,num_aug = 9):
    """
    load the annotations files, duplicates the ids, adding 9000000, add t to images names.
    Save the duplicated annotations in the new folder
    """
    # Full path for the data-file.
    path = os.path.join(annot_dir_in, "annotations/", filename_captions)
    path_out = os.path.join(annot_dir_out, "annotations/", filename_output)
    
    # Load the file .json.
    with open(path, "r", encoding="utf-8") as file:
        data_raw = json.load(file)
               
    # Convenience variables.
    # original lists of dictionnaries
    images = data_raw['images']
    annotations = data_raw['annotations']
        
    # Images
    images2 = [] # create a new list
    images0 = images.copy() # copy the original list
    
    for imagex in images0: # get each dictionnary
        for i in range(num_aug):
            current_value = i *1000000
            image0 = imagex.copy() # copy the dictionnary
            image0['id'] = str(int(image0['id']) + current_value) # modify ids
            image0['file_name'] = "t"+str(i)+"_" + image0['file_name']
            images2.append(image0) # fill the new list with dictionnaries
                   
    # Get the id and filename for this image.
    # gather original and new lists        
    images_db = images + images2
            
    # annotations
    annotations2 = []
    annotations0 = annotations.copy()

    for annotationx in annotations0:
        for i in range(num_aug):
            current_value = i *1000000
            annotation0 = annotationx.copy()
            annotation0['image_id'] = str(int(annotation0['image_id']) + current_value)
            annotation0['id'] = str(int(annotation0['id']) + current_value)
            annotations2.append(annotation0)
        
    annotations_db = annotations + annotations2
    
    # rebuilt the dictionnary
    data_out = data_raw.copy()
    data_out['images'] = images_db
    data_out['annotations'] = annotations_db
    
    with open(path_out, 'w') as f:
        json.dump(data_out, f)

    print(len(data_raw['images']) , len(data_raw['annotations']), 
          len(data_out['images']) , len(data_out['annotations']))     


# In[117]:


### Increase annotation file
annotations_duplication(annot_dir_in = "data/coco/", annot_dir_out="data/coco_new/", 
                        filename_captions = "captions_train2017.json", 
                         filename_output = "captions_train2017.json")


# # Reload the files

# In[118]:


# Dossier de téléchargement du dataset Microsoft Coco
coco.set_data_dir("data/coco_new/")

# Récupération des noms de fichiers (filenames) et des descriptions (captions) 
# du jeu d'entraînement (training set) à l'aide du module coco.py
# returns ids, filename, caption
# si pb ouvre dans notepad++ encoding UTF8, save as
_, filenames_train, captions_train = coco._load_records(train=True)


# In[119]:


# Vérification des structures de données # 118 287
num_images_train = len(filenames_train)
num_images_train


# In[120]:


len(captions_train)


# In[121]:


# group of all labels for each images
captions_train[273]


# In[122]:


# On obtient une liste de listes
num_captions_train = 0
min_caption_number = 10
max_caption_number = 0
for caption_list in captions_train:
    num_captions_train += len(caption_list)
    if len(caption_list) < min_caption_number:
        min_caption_number = len(caption_list)
    if len(caption_list) > max_caption_number:
        max_caption_number = len(caption_list)
print("Total caption number: " + str(num_captions_train))
print("Min caption number: " + str(min_caption_number))
print("Max caption number: " + str(max_caption_number))


# In[123]:


# Quantité moyenne de captions par image
mean_caption_number = num_captions_train / num_images_train
mean_caption_number


# ### continue with the notebook Image_Translator_MBscc

# In[ ]:




