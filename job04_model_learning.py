from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dropout, Dense, Flatten
import numpy as np

X_train = np.load('./crawling_data/news_data_X_train_max_14_wordsize_3127.npy', allow_pickle=True)
X_test = np.load('./crawling_data/news_data_X_test_max_14_wordsize_3127.npy', allow_pickle=True)
Y_train = np.load('./crawling_data/news_data_Y_train_max_14_wordsize_3127.npy', allow_pickle=True)
Y_test = np.load('./crawling_data/news_data_Y_test_max_14_wordsize_3127.npy', allow_pickle=True)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(3127, 300, input_length=14))
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.build(input_shape=(None, 14))  # 입력 데이터 크기 (None은 배치 크기)
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