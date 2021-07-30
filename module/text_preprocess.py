import numpy as np

def apply_model(x,model):
  try:
    return model[x]
  except:#임베딩이 안되는 경우 (리뷰없거나 등)
    return np.zeros(128) # 영벡터
    


def saved_model(x,model):
  try:
    return model[x]
  except:#임베딩이 안되는 경우 (리뷰없거나 등)
    return np.zeros(128) #임베딩 벡터 대신 영벡터
