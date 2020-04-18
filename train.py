'''
Training text generator model based on ANN Network.

04.07.20
YoonSoo NA
NAMZ Labs
'''

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 파라미터 값을 설정하기.
input_text = "C:/work/pj_sound/pj2/total/total_data.txt" # 읽어드릴 텍스트 데이터.
word_range = 3    # ANN 인풋으로 들어갈 단어 개수.
predict_range = 1 # ANN 아웃풋으로 나올 단어 개수.
hidden_units = 64 # hidden layer내 unit 개수.
batch_size = 32   # 훈련할 때 몇 덩이씩 묶어서 훈련에 넣을지 개수 정하기.
epoch = 10        # 총 훈련 도는 epoch 개수.
activation_function = 'relu' # activation 함수.

# 데이터 비율 나누기. 총 비율의 합은 1.0이 되어야 함.
train_ratio = 0.8 # 전체 데이터에서 train으로 사용할 데이터의 비율
val_ratio = 0.1 # 전체 데이터에서 val로 사용할 데이터의 비율
test_ratio = 0.1 # 전체 데이터에서 test로 사용할 데이터의 비율

''' mnist 데이터 셋이므로 무시합니다.
# Preprocess the data (these are Numpy arrays)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]
'''
# 단어를 onehot vector로 바꾸는 함수.
def word2onehot(word_list, word_dict):
    if type(word_list) is not list:      # is not: word_list에 type가 list가 아니면 true
        word_list = [word_list]          # if 문에서 나온 단어를 list화
    onehot_size = len(word_dict.keys())
    word_len = len(word_list)
    onehot = np.zeros([word_len, onehot_size], dtype="float16") # x=word_len, y=onehot_size
    for idx, word in enumerate(word_list):  # enumerate: 몇번째 for문인지 알수 있다
        onehot[idx, word_dict[word]] = 1    # idx: word_list에 인덱스 값 # onehot을 1로 통일 왜
    return onehot.flatten()     # 다차원 배열을 1차원 배열로 바꿔줌(one_hot vector이니까)
#텍스트와 텍스트는 딕셔너리 값을 넣으면 입력값을 원핫 백터로 변환해준다

# onehot vector를 단어로 바꾸는 함수.
# onehot vector로 들어간 input data 를 output으로 나올때 단어로 바꿔줌
def onehot2word(onehot_list, inverse_word_dict):
    onehot_len = len(onehot_list)
    word_len = len(inverse_word_dict.keys())
    word_range = int(onehot_len/word_len)
    new_onehot = onehot_list.reshape(word_range, word_len)
    recovered_word = []
    for idx in range(word_range):
        one_idx = int(np.where(new_onehot[idx]==1)[0])      # np.where 조건에 맞는 값을 찾는 함수
        # [0] 차원을 하나로 만듦
        recovered_word.append(inverse_word_dict[one_idx])
    return recovered_word
#원핫 백터를 이전에 딕셔너리값의 위치를 찾아서 이전 단어로 복귀 시켜준다


# text 읽어드리기.
print("데이터를 불러오는 중...")
with open(input_text,'r',encoding='utf-8') as txt:
    new_lines = txt.readlines()[0]      # 불러올때 차원이 하나 더생김 [0]을 붙이지 않으면 오류 발생 
    word_chunk = new_lines.split(' ')

# dict 만들기.
print("데이터로부터 dict 정보 추출하는 중...")
word_dict = {}
for w in word_chunk:
    if w not in word_dict.keys():   # 중복된 단어는 넣지 않는다
        word_dict[w] = len(word_dict)

# inverse dict 만들기.
inverse_word_dict = {v: k for k, v in word_dict.items()}  # .items() key and values 쌍을 리턴
#향상된 for문은 뒤에서 부터 읽으면 이해하기 쉬움 word_dict에서 k v를 불러오고 v, k로 순서 변경

# input과 output 데이터 생성하기.
print("input과 output 데이터 생성하는 중...")
word_num = len(word_chunk)
total_range = word_num - word_range - predict_range     # 전체에서 input과 output을 뺀 값

# 데이터 담을 리스트 미리 생성하기. (미리 생성해놓고 for loop을 돌면서 넣어야 속도가 느리지 않음.)
input_list = [None] * total_range   # total_range만큼의 [None]를 만들어 놓는다. 
#None을 해주지 않으면 total_range만큼의 길이로 만들 수 없음

output_list = [None] * total_range  # [](빈 리스트) * 1000은 [] [None]*1000 은 [None...None(1000개)]

for start_idx in range(total_range):
    if start_idx % 1000 == 0:       # 전체에서 1000씩 processing
        print("processing... {}/{}".format(start_idx, total_range))
    last_idx = start_idx + word_range
    input_list[start_idx] = word2onehot(word_chunk[start_idx:last_idx], word_dict)
    #array([0.0.1.0....],array([0.0.0... 같은 형식으로 되어있음
    output_list[start_idx] = word2onehot(word_chunk[last_idx], word_dict)

# list형식에서 numpy array 형식으로 바꾸기.
input_list = np.asarray(input_list)
#백터 형식이었던 input_list를 (x,y) 형식으로 바꿔줌
output_list = np.asarray(output_list)

# 모은 데이터를 train, val, test셋으로 나누기.
print("train, val, test용 데이터로 분리하는 중...")
input_num = len(input_list)
train_range = int(np.round(input_num * train_ratio))
val_range = train_range + int(np.round(input_num * val_ratio))

train_input = input_list[0:train_range]
train_output = output_list[0:train_range]
val_input = input_list[train_range:val_range]
val_output = output_list[train_range:val_range]
test_input = input_list[val_range:]
test_output = output_list[val_range:]

# ANN 네트워크 생성하기.
print("훈련용 ANN 네트워크 생성하는 중...")
input_dim = input_list.shape[1]
output_dim = output_list.shape[1]
#계산 해야할 onehot 값의 길아가 input_dim의 길이이므로 shape를 input_dim으로 넣는다
inputs = keras.Input(shape=(input_dim,), name='digits')
x = layers.Dense(hidden_units, activation=activation_function, name='dense_1')(inputs)
x = layers.Dense(hidden_units, activation=activation_function, name='dense_2')(x)
outputs = layers.Dense(output_dim, name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
              # Loss function to minimize
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              # List of metrics to monitor
              metrics=['categorical_accuracy'])

print('모델 훈련 시작!')
# 모델 훈련하기.
history = model.fit(train_input, train_output,
                    batch_size=batch_size,
                    epochs=epoch,
                    validation_data=(val_input, val_output))

# 훈련 결과.
print('\n모델 결과:', history.history)

# 훈련 모델을 이용하여 test 결과 추출하기.
print('\n훈련된 모델에 대한 test 결과 추출하는 중...')
results = model.evaluate(test_input, test_output, batch_size=128)
print('test loss, test acc:', results)

# 훈련 모델을 이용하여 test 단어 추출하기.
print('\n테스트 input으로 output 생성해보기.')
test_input_idx = 3 # test 단어 인덱스를 적어넣고 아래를 돌리면 인풋단어에 대한 아웃풋 단어 결과가 나옵니다.
predictions = model.predict(test_input[test_input_idx].reshape(1,len(test_input[test_input_idx])))
input_words = onehot2word(test_input[test_input_idx], inverse_word_dict)
output_word = inverse_word_dict[int(np.where(predictions[0] == predictions[0].max())[0])]
print('input 단어: {} / output 단어: {}'.format(input_words, output_word))
