import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from konlpy.tag import Okt #Kkma
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

df = pd. read_csv('./crawling_data/naver_headline_news_241223.csv')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head())
df.info()
print(df.category.value_counts())

X = df['titles']
Y = df['category']

# print(X[0])
# okt = Okt()
# okt_x = okt.morphs(X[0], stem=True)
# print('Okt :', okt_x)

# kkma = Kkma()
# kkma_x = kkma.morphs(X[0])
# print('kkma :', kkma_x)
# exit()
# Y = pd.get_dummies(Y)
# print(Y.head())

# encoder = LabelEncoder()
# labeled_y = encoder.fit_transform(Y)
# print(labeled_y[:3])

#가져오기
with open('./models/encoder.pickle', 'rb')as f:
    encoder = pickle.load(f)

label = encoder.classes_
print(label)

labeled_y = encoder.transform(Y)
onehot_Y = to_categorical(labeled_y)
print(onehot_Y)

okt = Okt()

for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
print(X)

stopwords = pd.read_csv('./crawling_data/stopwords.csv', index_col=0)
print(stopwords)
# 도움이 안되고 방해만 되는 부분을 모아둔 것 stopword

for sentence in range(len(X)):
    words = []
    for word in range(len(X[sentence])):
        if len(X[sentence][word]) > 1:
            if X[sentence][word] not in list(stopwords['stopword']):
                words.append(X[sentence][word])
    X[sentence] = ' '.join(words)

print(X[:5])

# 단어 하나하나에 라벨을 붙여줌 형태소를 라벨링하여 숫자로 바꿔줌
# 새로운 정보는 현재 학습에 의미학습이 안되있어서 0으로 바꿔줘야함

with open('./models/news_token.pickle', 'rb') as f:
    token = pickle.load(f)
tokened_X = token.texts_to_sequences(X)
print(tokened_X[:5])

for i in range(len(tokened_X)):
    if len(tokened_X[i]) > 19:
        tokened_X[i] = tokened_X[i][:19] #앞에서부터 자르고 싶으면 [-16:]
X_pad = pad_sequences(tokened_X, 19)
print(X_pad)
print(len(X_pad[0]))

model = load_model('./models/news_category_classfication_model_0.6842105388641357.h5')
preds = model.predict(X_pad)

predicts = []
for pred in preds:
    most = label[np.argmax(pred)]
    pred[np.argmax(pred)] = 0
    second = label[np.argmax(pred)]
    predicts.append([most, second])
df['predict'] = predicts

print(df.head(30))

score = model.evaluate(X_pad, onehot_Y)
print(score[1])

df['OX'] = 0
for i in range(len(df)):
    if df.loc[i, 'category'] != df.loc[i, 'predict']:
        df.loc[i, 'OX'] = 1

print(df.OX.mean())