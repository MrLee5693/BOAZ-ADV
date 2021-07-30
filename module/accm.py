#import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, BatchNormalization, Activation, Dropout, Concatenate, Multiply, Lambda, Reshape,Add
from tensorflow.keras import Model, Input, regularizers, initializers, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
from custom_layer import BiasLayer,bias_dropout,dropout

def build_model(args,num_user,num_item):
  user_id = Input(shape=(1,), name="User Index")
  user_embedding = Embedding(input_dim=num_user, output_dim=args.embed_dim,name="User_Embedding",activity_regularizer=regularizers.l2(args.l2), embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01))(user_id)
  user_embedding = Lambda(lambda x : x[:,0,:],name="User_Embedding2")(user_embedding)
  user_embedding = Lambda(lambda x : dropout(x,args.user_drop),name="User_Dropout")(user_embedding)
  user_bias = Embedding(input_dim=num_user, output_dim=1,name="User_Bias", embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01))(user_id)

  user_bias = Lambda(lambda x : bias_dropout(x, args.user_drop))(user_bias) # (batch, 1,1) -> Scaling
  user_bias = Reshape((-1,))(user_bias)

  item_id = Input(shape=(1,), name="Item Index")
  item_embedding = Embedding(input_dim=num_item, output_dim=args.embed_dim,name="Item_Embedding",activity_regularizer=regularizers.l2(args.l2), embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01))(item_id)
  item_embedding = Lambda(lambda x : x[:,0,:],name="Item_Embedding2")(item_embedding)
  item_embedding = Lambda(lambda x : dropout(x,args.item_drop),name="Item_Dropout")(item_embedding)
  item_bias = Embedding(input_dim=num_item, output_dim=1,name="Item_Bias", embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01))(item_id)

  item_bias = Lambda(lambda x : bias_dropout(x, args.item_drop))(item_bias) # (batch, embedding_dim)
  item_bias = Reshape((-1,))(item_bias)


  user = Add()([user_embedding,user_bias])
  item = Add()([item_embedding,item_bias])
  cf_mul = Multiply(name="CF_Multiply")([user,item])
  ReduceSum = Lambda(lambda z: K.sum(z, axis=1),name="CF_Prediction")
  cf_prediction = ReduceSum(cf_mul)


  user_gender = Input(shape = (3,), name="User Gender") # 사용자 성별
  user_favorite = Input(shape = (1,), name = "User Favorite or Not")
  user_survey = Input(shape=(5,),name="User Survey") # 리뷰 설문조사

  item_image = Input(shape=(228,228,3),name="Item Image") # 이미지 resize 228,228,3
  resnet = ResNet50(include_top=True, weights = "imagenet", input_tensor=item_image)
  image_feature = resnet.output # Resnet50 Finetuning

  item_review = Input(shape=(128,),name="Item Review") # 리뷰 벡터화

  image_feature_layer = Dense(args.feature_dim,name="Image_Feature",activity_regularizer=regularizers.l2(args.l2))(image_feature) # 이미지
  image_feature_layer = BatchNormalization(name="Image_BatchNorm")(image_feature_layer)
  image_feature_layer = Activation('relu',name="Image_ReLU")(image_feature_layer)
  image_feature_layer = Dropout(args.dropout,name="Image_Dropout")(image_feature_layer)

  review_feature_layer = Dense(args.feature_dim,name="Review_Feature",activity_regularizer=regularizers.l2(args.l2))(item_review) # 이미지
  review_feature_layer = BatchNormalization(name="Review_BatchNorm")(review_feature_layer)
  review_feature_layer = Activation('relu',name="Review_ReLU")(review_feature_layer)
  review_feature_layer = Dropout(args.dropout,name="Review_Dropout")(review_feature_layer)



  cb_concat = Concatenate(name="CB_Concat")([user_gender,user_favorite,user_survey,image_feature_layer,review_feature_layer])
  cb_prediction = Dense(1,name="CB_Prediction")(cb_concat)

  user_cf_cb = Concatenate(name="User_CF_CB")([user_embedding,user_gender,user_favorite,user_survey])
  item_cf_cb = Concatenate(name="Item_CF_CB")([item_embedding,image_feature_layer,review_feature_layer])

  # Attention Layer
  user_attention_probs  = Dense(user_cf_cb.shape[1], activation='softmax', name="User_Attention_Weight")(user_cf_cb)
  user_attention = Multiply(name="User_Elementwise_Mul")([user_cf_cb, user_attention_probs])

  item_attention_probs  = Dense(item_cf_cb.shape[1], activation='softmax',name="Item_Attention_Weight")(item_cf_cb)
  item_attention = Multiply(name="Item_Elementwise_Mul")([item_cf_cb, item_attention_probs])

  user_att = Dense(args.attention_size, name="User_ATT_layer")(user_attention)
  item_att = Dense(args.attention_size, name="Item_ATT_layer")(item_attention)

  att_mul = Multiply(name="Rating_Mul")([user_att, item_att])
  ReduceSum = Lambda(lambda z: K.sum(z, axis=1),name="ACCM_Prediction")
  accm_prediction = ReduceSum(att_mul)
  global_bias = BiasLayer(units=1)(accm_prediction)
  label_layer = Input(1)

  model = Model([user_id,item_id,user_gender,user_favorite,user_survey,item_image,item_review,label_layer],
                [cf_prediction,cb_prediction,accm_prediction,user_attention_probs,item_attention_probs])
  # ResNet50 Layer는 가중치 학습 X
  for layer in resnet.layers[:-26]:
    layer.trainable = False
  label = K.squeeze(label_layer,-1)
  cf_pred = cf_prediction
  cb_pred = K.squeeze(cb_prediction,-1)
  global_bias = global_bias
  user_bias = K.squeeze(user_bias,-1)
  item_bias = K.squeeze(item_bias,-1)
  accm_pred = accm_prediction
  not_cold_ratio = ((1 - args.user_drop) * (1 - args.item_drop))
  bias = global_bias + user_bias + item_bias
  cf_loss = K.sqrt(K.mean(K.square((cf_pred+bias -label)*not_cold_ratio ), axis=-1))
  cb_loss = K.sqrt(K.mean(K.square(cb_pred+bias -label), axis=-1))
  accm_loss = K.sqrt(K.mean(K.square(accm_pred-label), axis=-1))

  model.add_loss(cf_loss)
  model.add_loss(cb_loss)
  model.add_loss(accm_loss)

  return model