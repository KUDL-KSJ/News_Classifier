# -*- coding: utf-8 -*-

import torch 
import torch.nn as nn
import time
from torch.autograd import Variable
from TextCNN import TextCNN
from data_util import DataUtil
import json
import logging

#Hyper Parameters
seq_length = 750
num_classes = 4
embed_size = 128
filter_sizes = [1,2,3,4,5,6,7,8]
num_filters = 256
dropout_prob = 0.5
learning_rate = 1e-3
batch_size = 64
num_epochs = 5
vocab_size = 8000
log_window = 30
val_size = 7300 #4000    # validation size
make_model = False #True
make_test = True
categories =['politics','entertainment','sport','business']

model_path = 'models/textcnn_17-12-10_20-12-07.pkl'
train_path = 'train_ksj.json'
voca_path = 'voca.json'
val_path = 'val_ksj.json'
test_path = 'test_ksj.json'
savepath = ''

util = DataUtil(seq_length,vocab_size,batch_size,train_path,val_path,voca_path)
textcnn = TextCNN(seq_length, num_classes, vocab_size, embed_size,
                  filter_sizes, num_filters, dropout_prob)
textcnn.cuda()
# define loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(textcnn.parameters(), lr = learning_rate)
                                                                                
num_batches = util.num_batch


def val_accuracy(data):        # data = zip(input, label)
    
    textcnn.eval()
    num_correct = 0
    num_total = 0
    
    for input_data, label in data:
        inputs = Variable(torch.LongTensor(input_data)).cuda()
        label = torch.LongTensor(label).cuda()
        outputs = textcnn(inputs)
        _, predicted = torch.max(outputs.data, 1)
        num_total += outputs.size(0)
        num_correct += (label == predicted).sum()
        
    acc = float(num_correct) / num_total
    
    return acc


def test_accuracy(data):        # data = zip(input, label)
    
    textcnn.eval()
    num_correct = 0
    num_total = 0
    res= [[0]*num_classes for i in range(num_classes)]
    qq = []
    
    for input_data, label in data:
        inputs = Variable(torch.LongTensor(input_data)).cuda()
        label = torch.LongTensor(label).cuda()
        outputs = textcnn(inputs)
        _, predicted = torch.max(outputs.data, 1)
        num_total += outputs.size(0)
        num_correct += (label == predicted).sum()
        
        l = label.tolist()
        p = predicted.tolist()
        qq += p
        for ll,pp in zip(l,p):
            res[ll][pp] += 1
    
    acc = float(num_correct) / num_total
    
    return acc, qq, res

# train model
def train_model():
    
    logging.info("Make and Train model")
    step = 0
    
    for epoch in range(0, num_epochs):
        
        batch_ids_set, batch_labels_set = util.make_batch()
        #batch ids set = numofbatch * batchsize * seq_len
        
        textcnn.train() # Should we?
        
        for batch_input, batch_label in zip(batch_ids_set, batch_labels_set):
            # ids shape = batch_size x sequence length
            # get batch inputs and targets
            inputs = Variable(torch.LongTensor(batch_input), requires_grad=False).cuda()
            labels = Variable(torch.LongTensor(batch_label), requires_grad=False).cuda() # one-hot
            
            # Forward + Backward + Optimize
            textcnn.zero_grad()
            outputs = textcnn(inputs)
            loss = criterion(outputs, labels)
            optimizer = torch.optim.Adam(textcnn.parameters(), lr = learning_rate)
            loss.backward()
            for i in range(0,embed_size):
                textcnn.embed.weight.grad.data[0][i] = 0
            optimizer.step()
            
            step += 1
            if (step % log_window) == 0:
                print ('Epoch [%d/%d], Step[%d/%d], Train_Loss: %.3f' %
                       (epoch+1, num_epochs, step, num_batches*num_epochs, loss.data[0]))
        
        now = time.localtime()
        timestr = time.strftime("%y-%m-%d_%H-%M-%S", now)
        savepath = 'models/textcnn_'+timestr+'.pkl'
        torch.save(textcnn.state_dict(), savepath)
        
        val_data, val_labels = util.make_val(val_size)
        val_acc = val_accuracy(zip(val_data, val_labels))
        print('Epoch [%d/%d], Validation Accuracy: %.2f %%' % (epoch+1, num_epochs, val_acc*100.0))


if __name__ == "__main__":

    if make_model :
        st_time = time.time()
        train_model()
        print(time.time()-st_time)
    else : 
        logging.info("Load model")
        textcnn.load_state_dict(torch.load(model_path))

    # test
    
    if make_test:
        textcnn.eval()
        
        test_data, test_labels = util.testset_process(test_path)
        test_acc, predicted, res = test_accuracy(zip(test_data, test_labels))
        
        #print(predicted)
        print(res)
        for i in range(num_classes):
            print("acc of %s %.3f %%" %(categories[i], (float(res[i][i]) /sum(res[i]))*100.0 ))
            
        testlog = {'predicted' : predicted, 'res' : res}
        with open('output.json', 'w') as outfiles:
            json.dump(testlog,outfiles)
        
        print("Test Accuracy of TextCNN on test data : %.2f %%" %(test_acc*100.0))
