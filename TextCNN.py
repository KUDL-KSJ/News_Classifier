import torch 
import torch.nn as nn
import math

class TextCNN(nn.Module):
    def __init__(self, sequence_length, num_classes, vocab_size,
                embedding_size, filter_sizes, num_filters, dropout_prob):
        super(TextCNN, self).__init__()
        self.embed_size = embedding_size
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(1,num_filters, kernel_size=(filt_size, embedding_size)),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(),
                    nn.MaxPool2d((sequence_length-filt_size+1,1))
                ) for filt_size in filter_sizes ])

        self.drop_out = nn.Dropout(p=dropout_prob)
        self.BN = nn.BatchNorm1d(num_filters * len(filter_sizes))
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.init_weights()
    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1,0.1)
        for i in range(0, self.embed_size):
            self.embed.weight.data[0][i] = 0
        for conv in self.convs:
            for layer in conv:
                if type(layer) == nn.Conv2d:
                    torch.nn.init.xavier_normal(layer.weight)
                    layer.weight.data *= math.sqrt(2)
                elif type(layer) == nn.BatchNorm2d:
                    layer.weight.data.normal_(1.0, 0.02)
        torch.nn.init.xavier_normal(self.fc.weight)
        
    def forward(self, x):
        out = self.embed(x)
        
        #print(out.shape)
        out = torch.unsqueeze(out,1)
        #logging.info(out.size)
        conv_out = []
        for conv in self.convs:
            conv_out.append(conv(out))
        out = torch.cat(conv_out, 1)
        out = self.drop_out(out)
        out = self.BN(out)
        out = torch.squeeze(out,3)
        out = torch.squeeze(out,2)
        out = self.fc(out)
        return out
        