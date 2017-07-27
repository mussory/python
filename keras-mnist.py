from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout ,Activation
from keras.optimizers import Adam
from keras.utils import np_utils

# MNISTのデータを読み込む
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# データをfloat32型に変換して正規化する
X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float')
X_train /= 255
X_test /= 255
# ラベルデータを0-9までのカテゴリをあらわす配列に変換
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

#モデルの構造を定義
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

#モデルを構築
model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'])
# データで訓練
hist = model.fit(X_train, y_train)

#テストデータで訓練
score = model.evaluate(X_train, y_train, varbose=1)
print('loss=', score[0])
print('accuracy=',score[1])    
