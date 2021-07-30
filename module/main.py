import os #for accessing the file system of the system
#import random
from gensim.models import FastText
from konlpy.tag import Okt
import re
import pandas as pd
#import keras
import numpy as np
import json
import pickle
import warnings
from keras.callbacks import ModelCheckpoint
import argparse
import wandb
from wandb.keras import WandbCallback
from DataGenerator import DataGenerator
from text_preprocess import apply_model,saved_model
from accm import build_model
from tensorflow.keras import optimizers
import tensorflow as tf
from custom_layer import BiasLayer
import ast
warnings.filterwarnings("ignore")


def main():
  wandb.init(project= "ACCM")
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', type=int, default=256,
                      help='input batch size')
  parser.add_argument('--lr', type=float, default=1e-3,
                      help='learning rate')
  parser.add_argument('--epochs', type=int, default=5,
                      help='training epochs')
  parser.add_argument('--embed_dim', type=int, default=64,
                      help='User and Item Embedding Dimension')
  parser.add_argument('--feature_dim', type=int, default=64,
                      help='Image and Review Dimension')
  parser.add_argument('--attention_size', type=int, default=16,
                      help='attention size for ACCM')
  parser.add_argument('--dropout', type=float, default=0.2,
                      help='dropout for Image and Review Layer')
  parser.add_argument('--user_drop', type=float, default=0.5,
                      help='User Dropout Rate')
  parser.add_argument('--item_drop', type=float, default=0.5,
                      help='Item Dropout Rate')
  parser.add_argument('--l2', type=float, default=1e-4,
                      help='L2 Regularization')
  parser.add_argument('--img_path', type=str, default='./data/image',
                      help='Saved Image Path')
  parser.add_argument('--img_size', type=int, default=228,
                      help='Image Resize for Resnet50')
  parser.add_argument('--img_train', 
                      help='Image Train', action='store_false')
  parser.add_argument('--review_train', 
                      help='Review Train', action='store_false')
  args = parser.parse_args()
  wandb.config.update(args)
  print(args)
  print("Data Load ...")
  train = pd.read_csv("./data/train_integration.csv",index_col=0)
  test = pd.read_csv("./data/test_integration.csv",index_col=0)
  num_user = train["UserID"].nunique()+1000
  num_cafe = train["ItemID"].nunique()+1
  if (args.review_train) & (args.img_train):
    if os.path.isfile('./data/fasttext_model'):
      #save_model = FastText.load('./data/fasttext_model')
      
      #train['Review_vector'] = train['Review'].apply(lambda x : saved_model(x))
      #test['Review_vector'] = test['Review'].apply(lambda x : saved_model(x))
      review = pd.read_csv("./data/review.csv")
    
      review["Review_vector"] = review["Review_vector"].apply(lambda x : x.replace("\n",""))
      review["Review_vector"] = review["Review_vector"].apply(lambda x :" ".join(x.split()))
      review["Review_vector"] = review["Review_vector"].apply(lambda x :x.replace("[ ","["))
      review["Review_vector"] = review["Review_vector"].apply(lambda x : x.replace(" ]","]"))
      review["Review_vector"] = review["Review_vector"].apply(lambda x : x.replace(" ",","))
      review["Review_vector"] = review["Review_vector"].apply(lambda x : ast.literal_eval(x))
    else:
      review = pd.read_csv("./data/review.csv")
      okt = Okt() 
      review['Review'] = review["Review"].apply(lambda x : re.sub(pattern = '([ㄱ-ㅎㅏ-ㅣ]+)',repl=' ',string=str(x)) )
      review["Review_vector"] = review["Review"].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
      review['Review_vector'] = review['Review_vector'].apply(lambda x : okt.morphs(x))
      ft_model = FastText(sentences = list(review['Review_vector'].values), size=128, window=3, min_count=1) #변환되는 벡터 크기 : (size, )
      review['Review_vector'] = review['Review_vector'].apply(lambda x : apply_model(x))
      #train["Review"] = train["Review"].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
      #train['Review_token'] = train['Review'].apply(lambda x : okt.morphs(x))
      #ft_model = FastText(sentences = list(train['Review_token'].values), size=128, window=3, min_count=1) #변환되는 벡터 크기 : (size, )
      #train['Review_vector'] = train['Review'].apply(lambda x : apply_model(x))
      #test['Review_vector'] = test['Review'].apply(lambda x : apply_model(x))
      ft_model.save('./data/fasttext_model')
    # train, test Datagenerator 클래스를  각각 생성합니다.
    train_gen = DataGenerator(data=train, imgs_dir=args.img_path, img_size=args.img_size, batch_size=args.batch_size,img_train=True,review_train=True,review=review)
    test_gen = DataGenerator(data=test, imgs_dir=args.img_path, img_size=args.img_size, batch_size=args.batch_size,img_train=True,review_train=True,review=review)
  elif args.img_train:
    # train, test Datagenerator 클래스를  각각 생성합니다.
    train_gen = DataGenerator(data=train, imgs_dir=args.img_path, img_size=args.img_size, batch_size=args.batch_size,img_train=True,review_train=False)
    test_gen = DataGenerator(data=test, imgs_dir=args.img_path, img_size=args.img_size, batch_size=args.batch_size,img_train=True,review_train=False)
  
  elif args.review_train:
    if os.path.isfile('./data/fasttext_model'):
      #save_model = FastText.load('./data/fasttext_model')
      
      #train['Review_vector'] = train['Review'].apply(lambda x : saved_model(x))
      #test['Review_vector'] = test['Review'].apply(lambda x : saved_model(x))
      review = pd.read_csv("./data/review.csv")
      review["Review_vector"] = review["Review_vector"].apply(lambda x : x.replace("\n",""))
      review["Review_vector"] = review["Review_vector"].apply(lambda x :" ".join(x.split()))
      review["Review_vector"] = review["Review_vector"].apply(lambda x :x.replace("[ ","["))
      review["Review_vector"] = review["Review_vector"].apply(lambda x : x.replace(" ]","]"))
      review["Review_vector"] = review["Review_vector"].apply(lambda x : x.replace(" ",","))
      review["Review_vector"] = review["Review_vector"].apply(lambda x : ast.literal_eval(x))
    else:
      review = pd.read_csv("./data/review.csv")
      okt = Okt() 
      review['Review'] = review["Review"].apply(lambda x : re.sub(pattern = '([ㄱ-ㅎㅏ-ㅣ]+)',repl=' ',string=str(x)) )
      review["Review_vector"] = review["Review"].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
      review['Review_vector'] = review['Review_vector'].apply(lambda x : okt.morphs(x))
      ft_model = FastText(sentences = list(review['Review_vector'].values), size=128, window=3, min_count=1) #변환되는 벡터 크기 : (size, )
      review['Review_vector'] = review['Review_vector'].apply(lambda x : apply_model(x))
      #train["Review"] = train["Review"].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
      #train['Review_token'] = train['Review'].apply(lambda x : okt.morphs(x))
      #ft_model = FastText(sentences = list(train['Review_token'].values), size=128, window=3, min_count=1) #변환되는 벡터 크기 : (size, )
      #train['Review_vector'] = train['Review'].apply(lambda x : apply_model(x))
      #test['Review_vector'] = test['Review'].apply(lambda x : apply_model(x))
      ft_model.save('./data/fasttext_model')
    # train, test Datagenerator 클래스를  각각 생성합니다.
    train_gen = DataGenerator(data=train, imgs_dir=args.img_path, img_size=args.img_size, batch_size=args.batch_size,img_train=False,review_train=True,review=review)
    test_gen = DataGenerator(data=test, imgs_dir=args.img_path, img_size=args.img_size, batch_size=args.batch_size,img_train=False,review_train=True,review=review)
  else:
    # train, test Datagenerator 클래스를  각각 생성합니다.
    train_gen = DataGenerator(data=train, imgs_dir=args.img_path, img_size=args.img_size, batch_size=args.batch_size,img_train=False,review_train=False)
    test_gen = DataGenerator(data=test, imgs_dir=args.img_path, img_size=args.img_size, batch_size=args.batch_size,img_train=False,review_train=False)

  model = build_model(args,num_user,num_cafe)

 

  print("total training batches: ", len(train_gen))
  print("total validaton batches: ", len(test_gen))
  train_steps = len(train) // args.batch_size
  test_steps = len(test) // args.batch_size


  # define model
  optimizer = optimizers.Adam(lr=args.lr, decay=1e-5)
  model.compile(optimizer=optimizer)

  # fit model
  model.fit_generator(generator=train_gen,
                    steps_per_epoch=train_steps,validation_data=test_gen,
                    validation_steps=test_steps,
                    epochs=args.epochs,
                    callbacks = [WandbCallback()])
  config = model.get_config()
  ma = f"./Result/Model Architecture_image_{args.img_train}_review_{args.review_train}_E{args.embed_dim}dim_F{args.feature_dim}dim_Att{args.attention_size}dim_D{args.user_drop}_l2{args.l2}.pickle"
  with open(ma,"wb") as p_file:
    pickle.dump(config,p_file)
  with open(ma,"rb") as p_file:
    config2 = pickle.load(p_file)
  new_model = tf.keras.Model.from_config(config2,custom_objects={"BiasLayer":BiasLayer})
  save_file = f'./Result/ACCM_image_{args.img_train}_review_{args.review_train}_E{args.embed_dim}dim_F{args.feature_dim}dim_Att{args.attention_size}dim_D{args.user_drop}_l2{args.l2}.h5'
  model.save_weights(save_file)
  
  new_model.load_weights(save_file)
 

  
  #print(config)
if __name__ == "__main__":
  main()
