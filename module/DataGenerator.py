import tensorflow as tf
import pandas as pd
import numpy as np
import os
from skimage import io
from skimage.transform import resize


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, imgs_dir, batch_size=128, img_size=224, review_size=128,shuffle=True, img_train=False,review_train=False,review=None):
        self.data = data
        self.indexes = list(self.data.index)
        self.imgs_dir = imgs_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.review_size = review_size
        self.shuffle = shuffle
        self.img_train = img_train

        self.review_train = review_train
        self.review = review
        self.on_epoch_end()

    # for printing the statistics of the function
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation__(self, id_name):
        'Generates data containing batch_size samples'
        # Initialization 
        if (self.img_train) | (self.review_train):
          user = self.data.loc[id_name,"UserID"]
          item = self.data.loc[id_name,"ItemID"]
          time = self.data.loc[id_name,"Timestamp"]
          data_list = self.data[(self.data["Timestamp"] < time) & (self.data["ItemID"] == item)]
          
        
        
        user_id = self.data.loc[id_name,"UserID_Re"]
        item_id = self.data.loc[id_name,"ItemID_Re"]
        target = self.data.loc[id_name,"Rating"]
        if self.img_train:
          #if self.data.loc[id_name,"Image"] != "이미지 없음":
          try:
            data_list = data_list.sample(n=1).reset_index(drop=True)
            img_path = os.path.join(self.imgs_dir, f"{data_list['ItemID']}_{data_list['ReviewID']}.jpg") # 이미지 1개 경로
            img = io.imread(img_path)
            img = resize(img, (self.img_size, self.img_size))
            # image normalization
            img = img / 255.0
          except:
            img = np.zeros((self.img_size,self.img_size,3)) # 이미지 사용 X
          #else:
          #  img = np.zeros((self.img_size,self.img_size,3)) # 이미지 사용 X
        else:
          img = np.zeros((self.img_size,self.img_size,3)) # 이미지 사용 X

        if self.review_train:
          try:
            data_list = data_list.sample(n=1).reset_index(drop=True)
            review_vector = self.review[(self.review["ReviewID"]==data_list['ReviewID']) & (self.review["ItemID"]==data_list['ItemID'])]["Review_vector"]
          except:
            review_vector = np.zeros(self.review_size, dtype="float")
        else:
          review_vector = np.zeros(self.review_size, dtype="float")

        # 8:11 : 성별, 11 : 좋아요, 12:17
        gender = self.data.iloc[id_name,6:9].astype("float")
        favorite = self.data.iloc[id_name,9].astype("float")
        survey = self.data.iloc[id_name,10:15].astype("float")
  
        return user_id,item_id,gender,favorite,survey,img,review_vector,target




    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.data) / self.batch_size))
    def __getitem__(self, index):  # index : batch no.
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_ids = indexes
        
        user_ids = []
        item_ids = []
        imgs = []
        reviews = []
        genders = []
        favorites = []
        surveys = []
        trues = []

        for id_name in batch_ids:
            user,item,gender,favorite,survey,img,review,true= self.__data_generation__(id_name)
            user = user.reshape(-1)
            item = item.reshape(-1)
            favorite =  favorite.reshape(-1)
            true = true.reshape(-1)
            user_ids.append(user)
            item_ids.append(item)
            imgs.append(img)
            reviews.append(review)
            genders.append(gender)
            favorites.append(favorite)
            surveys.append(survey)
            trues.append(true)

        user_ids = np.array(user_ids)
        item_ids = np.array(item_ids)
        imgs = np.array(imgs)
        reviews = np.array(reviews)
        genders = np.array(genders)
        favorites = np.array(favorites)
        surveys = np.array(surveys)
        trues = np.array(trues)
        
        y_batch = trues
        X_batch = [user_ids,item_ids,genders,favorites,surveys,imgs,reviews,y_batch]
        
        return X_batch