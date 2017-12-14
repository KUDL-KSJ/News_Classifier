# News_Classifier
News article category discriminator​ for KUDL2017 term project

컴퓨터학과
2015410062 Shin Gi Cheol​
2015410066 Kim Jun Ha​
2015410109 Jeong Seung Yeon​
가 개발하였습니다.

학습 및 test는
train_path = 'train_ksj.json'
voca_path = 'voca.json'
val_path = 'val_ksj.json'
test_path = 'test_ksj.json'
로 지정후 진행합니다.

model load시 make_model = False로 지정한 후,
model_path로 load할 model path를 지정합니다.

voca.json이 존재하면 dictionary 파일을 새로 만들지 않으므로 
seq_length, voca_size를 변경할 때마다 꼭 기존의 voca.json파일을 지운 후에 실행시켜주셔야 합니다.