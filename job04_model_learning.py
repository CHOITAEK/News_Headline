# from matplotlib import pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dropout, Dense, Flatten
# import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import glob

from keras._tf_keras.keras.models import *
from keras._tf_keras.keras.layers import *
from keras._tf_keras.keras.callbacks import *
from keras._tf_keras.keras.optimizers import Adam

X_train = np.load('./crawling_data/news_data_X_train_wordsize_6666_max19.npy', allow_pickle=True)
X_test = np.load('./crawling_data/news_data_X_test_wordsize_6666_max19.npy', allow_pickle=True)
Y_train = np.load('./crawling_data/news_data_Y_train_wordsize_6666_max19.npy', allow_pickle=True)
Y_test = np.load('./crawling_data/news_data_Y_test_wordsize_6666_max19.npy', allow_pickle=True)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(6666, 300)) #각각의 형태소 공간을 벡터화 해주는 레이어가 임베딩 레이어
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
# 컨브 레이어는 순서 학습 안됨 앞뒤 관계는 학습이 됨
model.add(MaxPooling1D(pool_size=1)) #conv 따라감

model.add(LSTM(128, activation='tanh', return_sequences=True))
#rnn은 입력이 2개  LSTM은 activation이 tanh
model.add(Dropout(0.35))

model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.35))

model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.35))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(6, activation='softmax'))

model.build(input_shape=(None, 19))  # 입력 데이터 크기 (None은 배치 크기)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=128,
                     epochs=10, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('final test set accuracy', score[1])
model.save('./models/news_category_classfication_model_{}.h5'.format(
    fit_hist.history['val_accuracy'][-1]))
plt.plot(fit_hist.history['val_accuracy'], label='val_accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()