#from keras.preprocessing.image import load_img,img_to_array
from keras.models import Model
#  preprecessing of images -  including normalization of  color index ,  rescaling of image color indexes
from keras.applications.imagenet_utils import preprocess_input

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import jovian
import sys
import shutil

from keras.applications import (vgg16)                  
#from keras.applications import (vgg16)

#nb_closest_images = 2  number of most similar images to retrieve

import warnings


def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    

import os
"""Loaded Successfully"""
  
source = input("Enter Image Name: ")
src = "C:/Users/Arrow/Downloads/"+source
  
destination = "E:/project1/pictures/"+source
 
dest = shutil.copyfile(src, destination) 

class ImageRecommender : 
    
    def __init__(self, model, list_of_image, filespath) :
        #  models  can be VGG16
    
        # file -  list of images 
        
        #  prepare  the raw  model  ready  along with  our give  data 
        
        self.model = model
        self.filespath = filespath
        self.list_of_image = list_of_image
        #since ouput.shape return object dimension just eval it to get integer ...
        self.image_width = eval(str(self.model.layers[0].output.shape[1]))
        self.image_height = eval(str(self.model.layers[0].output.shape[2]))
        # remove the last two layers in order to get features instead of predictions
        self.image_features_extractor = Model(inputs=self.model.input, 
                                              outputs=self.model.layers[-2].output)
        self.processed_image = self.Pics2Matrix()
        
        self.sim_table = self.GetSimilarity(self.processed_image)
        
    #   load the  image  and  reszie  to the  model image size    
    def ddl_images(self, image_url) :
        try : 
            return load_img(self.filespath + image_url, 
                            target_size=(self.image_width, self.image_height))
        except OSError : 
            # image unreadable // remove from list
            self.list_of_image = [x for x in self.list_of_image if x != image_url]
            #self.list_of_image.remove(image_url)
            pass
    
   
    
    #  image has  4  dimensional (R G B  and  channel =3,   if  RGG, obacity then chanenl = 4   )
    #  convert  for example  50x50 ,  with  3 color channel (RBG) then it will be converted to  one dimension of size  =  50*50*3
    #  so that  to predict / process / apply the model
    
    def Pics2Matrix(self) :

        # convert the PIL image to a numpy array
        # in PIL - image is in (width, height, channel)
        # in Numpy - image is in (height, width, channel)
        # convert the image / images into batch format
        # expand_dims will add an extra dimension to the data at a particular axis
        # we want the input matrix to the network to be of the form (batchsize, height, width, channels)
        # thus we add the extra dimension to the axis 0.
        #from keras.preprocessing.image import load_img,img_to_array
        
        list_of_expanded_array = list()
        for i in tqdm(range(len(self.list_of_image) - 1)) :
            try :
                tmp = img_to_array(self.ddl_images(self.list_of_image[i]))
                expand = np.expand_dims(tmp, axis = 0)
                # expand  is  one dimensional  view  of image 
                list_of_expanded_array.append(expand)
                #  list of one dimensonal  iamge  of 253 size
            except ValueError : 
                self.list_of_image = [x for x in self.list_of_image if x != self.list_of_image[i]]
                #self.list_of_image.remove(self.list_of_image[i])
        images = np.vstack(list_of_expanded_array)
        
        return preprocess_input(images)
    
    #  makign compariosn of  given image with the dataset of 253 iamges
    def GetSimilarity(self, processed_imgs) :
       
        imgs_features = self.image_features_extractor.predict(processed_imgs)
       
        cosSimilarities = cosine_similarity(imgs_features)
        cos_similarities_df = pd.DataFrame(cosSimilarities, 
                                           columns=self.list_of_image[:len(self.list_of_image) -1],
                                           index=self.list_of_image[:len(self.list_of_image) -1])
        return cos_similarities_df
    

    # This will recommend most closest  three images 
    def most_similar_to(self, given_img, nb_closest_images = 3):

        print("-----------------------------------------------------------------------")
        print("Original Shirt:")
        
        original = self.ddl_images(given_img)
        plt.imshow(original)
        plt.show()

        print("-----------------------------------------------------------------------")
        print("Most similar Shirt:")
        print(given_img)
        
        closest_imgs = self.sim_table[given_img].sort_values(ascending=False)[1:nb_closest_images+1].index
        closest_imgs_scores = self.sim_table[given_img].sort_values(ascending=False)[1:nb_closest_images+1]
       
        for i in range(0,len(closest_imgs)):
            original = self.ddl_images(closest_imgs[i])
            plt.imshow(original)
            plt.show()
            print("similarity score : ",closest_imgs_scores[i])
# loading 2 models in the program
#  weights  can be  other  mnist , famnist (?), 

vgg_model = vgg16.VGG16(weights='imagenet')
%matplotlib inline 
from tqdm import tqdm

list_of_pretrained = [
                      vgg_model                    
                      ]



  
 
for pretrained_model in list_of_pretrained : 
    print('=========================================')
    print('trained model %s are running' %pretrained_model)
    print('=========================================')
    pretrained_recommender = ImageRecommender(pretrained_model, files, filespath = 'E:\project1\pictures\\')
   
    Upload_IMG= files.index(source)
    pretrained_recommender.most_similar_to(files[Upload_IMG])
    print('\n')
