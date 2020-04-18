'''
텍스트 데이터 정제기

2020.04.10
'''

import glob
import re
#데이터에 s(start)와 e(end)를 붙여준 문서
sentence_list=[]
#문장 불러오기
for filename in glob.glob('c:/work/task4/audio_data/HSA/*/*.txt'):
    text=open(filename,'r',encoding='utf-8')
    text_list =text.readlines()
#\n 제거 밑 s와 e 입력하기
    li=[]
    for i in text_list:
        sub_n=re.sub('\n*','',i)
        add_se='s '+sub_n+' e '+'\n'
        li.append(add_se)
#str로 변경
    change_str=''.join(li)
#연속으로 나오는 줄바꿈 제거
    make_space_same=re.sub('\n\n','\n',change_str)
#결과값 리스트에 입력
    sentence_list.append(make_space_same)
#예외사항 정리하기
join_read=''.join(sentence_list)
sub_n=re.sub('\n *',' ',join_read)
sub_s=re.sub('s *','s ',sub_n)
sub_e=re.sub(' *e',' e',sub_s)
sub_es=re.sub('e *s','e s',sub_e)
sub_ese=re.sub('e s e','e',sub_es)
bbb = open('c:/work/task4/hantesta.txt', 'w', encoding='utf-8')
bbb.write(sub_ese)