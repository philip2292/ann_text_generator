'''

ANN 네트워크를 이용한 text generator 훈련 모델
2020.04.10
2조
나윤수 박다현 오상혁 지자현
'''
#라이브러리 목록
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import time
#파라미터 값을 미리 설정
input_text = "hantesth.txt"  # 읽어드릴 텍스트 데이터.
word_range = 3               # ANN 인풋으로 들어갈 단어 개수.
predict_range = 1            # ANN 아웃풋으로 나올 단어 개수.
hidden_units = 64            # hidden layer내 unit 개수.
batch_size = 32              # 훈련할 때 몇 덩이씩 묶어서 훈련에 넣을지 개수 정하기.
epoch = 1                 # 총 훈련 도는 epoch 개수.
activation_function = 'relu' # activation 함수.


#단어를 원핫 인코딩하는 함수
def word2onehot(word_list,word_dict):
    #데이터가 리스트 형식이 아니면 리스트로 변경함
    if type(word_list) is not list:
        word_list = [word_list]
    #원핫 인코딩 모양 만들기
    onehot_size= len(word_dict.keys())
    word_len=len(word_list)
    onehot=np.zeros([word_len, onehot_size],dtype='float32')
    #for문으로 key값과 value값 받기
    for i , w in enumerate(word_list):
        onehot[i,word_dict[w]]=1
    return onehot.flatten()

#원핫백터가 나옴 [0.1.]

#원 핫 백터를 다시 단어로 바꿔주는 함수
def onehot2word(onehot_list, inverse_word_dict):

    #단어의 위치를 찾아내는 함수
    onehot_len=len(onehot_list)
    word_len = len(inverse_word_dict.keys())
    word_range = int(onehot_len/word_len)

    #원핫백터의 모양 변경
    new_onehot = onehot_list.reshape(word_range,word_len)

    #원핫 백터로 바꾼 단어를 재생
    recovered_word=[]
    for i in range(word_range):
        one_i=int(np.where(new_onehot[i]==1)[0])
        recovered_word.append(inverse_word_dict[one_i])

    return recovered_word
#원래 단어의 위치를 찾아서 나옴

#문서 불러오기
print('문서를 불러오는 중')
t1=time.time()
with open(input_text,'r',encoding='utf-8') as text:
    read_text = text.readlines()[0] #[0]이유 한번 [1]로 print해보면 이해 쉬움


    #데이터를 띄어쓰기 단위로 나누기
    word_split=read_text.split(' ')
#띄어쓰기 단위로 나뉘어 나옴
#문서를 딕셔너리 형태로 변경
voca_dict={}
for w in word_split:
    if w not in voca_dict.keys():
        voca_dict[w] = len(voca_dict)

#{'s': 0, '그때': 1, '저는': 2, '이': 3, '헤어': 4, '워터를': 5,......

#딕셔너리의 키값과 벨류값을 교환
change_key_value = {v: k for k, v in voca_dict.items()}
#{0: 's', 1: '그때', 2: '저는', 3: '이', 4: '헤어', 5: '워터를', 6: '뿌려주는데요', 7: 'e', 8: '

t2=time.time()
print('소요 시간',round(t2-t1,2),'초')
#입력층과 출력층 생성

t3=time.time()
print('입력층 출력층 데이터 생성')
word_num=len(word_split)
total_range= word_num - word_range - predict_range
#3622 total_range

#입력층 출력층 리스트 생성
list_in=[None]*total_range  # 리스트 요소를 3622개로 만들어줌
list_out=[None]*total_range
for start_idx in range(total_range):
 #진행 사항 출력
    if start_idx % 1000 == 0:
        print("processing... {}/{}".format(start_idx, total_range))
    #미리 만들어 놓은 입력층 출력층에 데이터 입력
    last_idx = start_idx + word_range
    list_in[start_idx] = word2onehot(word_split[start_idx:last_idx], voca_dict)
    #array([0.0.1.0....],array([0.0.0... 같은 형식
    list_out[start_idx] = word2onehot(word_split[last_idx], voca_dict)


#np.array형식으로 변환

list_in=np.asarray(list_in)
#(3622,4200) 형식으로 변함

list_out=np.asarray(list_out)
#위와 동일

# train valid test 데이터 비율 8:1:1로 나누기.
# 총 비율의 합은 1.0이 되어야 함.
eighty_per=int(len(list_in)*0.8)
ninety_per=int(len(list_in)*0.9)
in_train,out_train=list_in[:eighty_per],list_out[:eighty_per]
in_valid,out_valid=list_in[eighty_per:ninety_per],list_out[eighty_per:ninety_per]
in_test,out_test=list_in[ninety_per:],list_out[ninety_per:]
t4=time.time()
print('소요 시간',round(t4-t3,2),'초')
# ANN 네트워크 생성하기.
print("훈련용 ANN 네트워크 생성하는 중...")
t5=time.time()
#입력층 출력층 차원 설정
input_dim = list_in.shape[1]


output_dim = list_out.shape[1]
#입력층 히든레이어 출력층 설정
inputs = keras.Input(shape=(input_dim,), name='digits')
#구분해야할 onehot 백터의 길이가 4682이므로
x = layers.Dense(hidden_units, activation=activation_function, name='dense_1')(inputs)
x = layers.Dense(hidden_units, activation=activation_function, name='dense_2')(x)
outputs = layers.Dense(output_dim, name='predictions')(x)
#모델 생성
model = keras.Model(inputs=inputs, outputs=outputs)
#모델에 사용할 옵티마이저 로스펑션 메트릭스 등 설정
model.compile(optimizer=keras.optimizers.RMSprop(),  # 옵티마이저 설정
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),  # 손실함수
              metrics=['categorical_accuracy'])  # 관측 할 메트릭스 목록
t6=time.time()
print('소요 시간',round(t6-t5,2),'초')
print('모델 훈련 시작!')
t7=time.time()
# 모델 훈련하기.
#batch size epoch값 등 훈련 모델 설정
history = model.fit(in_train, out_train,
                    batch_size=batch_size,
                    epochs=epoch,
                    validation_data=(in_valid, out_valid))

# 훈련 결과 출력.
print('\n모델 결과:', history.history)
# 훈련 모델을 이용하여 test 결과 추출하기.
print('\n훈련된 모델에 대한 test 결과 추출하는 중...')
results = model.evaluate(in_test, out_test, batch_size=128)
#train set을 test set과 비교
print('test loss, test acc:', results)

t8=time.time()
print('소요 시간',round(t8-t7,2),'초')
# 훈련 모델을 이용하여 test 단어 추출하기.
print('\n테스트 input으로 output 생성해보기.\n * S: 문장 시작 심볼 E: 문장 종료 심볼 ')

test_input_idx =13 # test 단어 인덱스를 적어넣고 아래를 돌리면 인풋단어에 대한 아웃풋 단어 결과가 나옵니다.
predictions = model.predict(in_test[test_input_idx].reshape(1,len(in_test[test_input_idx])))
print(len(in_test[test_input_idx]))
print(predictions)
input_words = onehot2word(in_test[test_input_idx], change_key_value)
output_word = change_key_value[int(np.where(predictions[0] == predictions[0].max())[0])]
print('input 단어: {} / output 단어: {}'.format(input_words, output_word))

