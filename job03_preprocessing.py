import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from konlpy.tag import Okt #Kkma
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


df = pd. read_csv('./crawling_data/naver_headline_news_4_5_241219.csv')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head())
df.info()
print(df.category.value_counts())

X = df['titles']
Y = df['category']

print(X[0])
okt = Okt()
okt_x = okt.morphs(X[0], stem=True)
print('Okt :', okt_x)

# kkma = Kkma()
# kkma_x = kkma.morphs(X[0])
# print('kkma :', kkma_x)
# exit()
# Y = pd.get_dummies(Y)
# print(Y.head())

encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)
print(labeled_y[:3])

label = encoder.classes_
print(label)

with open('./models/encoder.pickle', 'wb')as f:
    pickle.dump(encoder, f)

onehot_Y = to_categorical(labeled_y)
print(onehot_Y)


for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
print(X)

stopwords = pd.read_csv('./crawling_data/stopwords.csv', index_col=0)
print(stopwords)


for sentence in range(len(X)):
    words = []
    for word in range(len(X[sentence])):
        if len(X[sentence][word]) > 1:
            if X[sentence][word] not in list(stopwords['stopword']):
                words.append(X[sentence][word])
    X[sentence] = ' '.join(words)

print(X[:5])

# 단어 하나하나에 라벨을 붙여줌
token = Tokenizer()
token.fit_on_texts(X) # low list 만든것
tokened_X = token.texts_to_sequences(X) #list 형태로 만들어 주는 부분
wordsize = len(token.word_index) + 1
print(wordsize)

print(tokened_X[:5])

# 긴거에 맞추어서 잛은 것에 0을 주어 맞춤

#최대값 찾기 알고리즘

max = 0

for i in range(len(tokened_X)):
    if max < len(tokened_X[i]):
        max = len(tokened_X[i])
print(max)


X_pad = pad_sequences(tokened_X, max)
print(X_pad)
print(len(X_pad[0]))

X_train, X_test, Y_train, Y_test = train_test_split(X_pad, onehot_Y, test_size=0.1)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

np.save('./crawling_data/news_data_X_train_max_{}_wordsize_{}'.format(max, wordsize), X_train)
np.save('./crawling_data/news_data_Y_train_max_{}_wordsize_{}'.format(max, wordsize), Y_train)
np.save('./crawling_data/news_data_X_test_max_{}_wordsize_{}'.format(max, wordsize), X_test)
np.save('./crawling_data/news_data_Y_test_max_{}_wordsize_{}'.format(max, wordsize), Y_test)
