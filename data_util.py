import re
import json
import math
from random import shuffle 
from nltk import PorterStemmer
from nltk.corpus import stopwords
from torch.utils.data.sampler import BatchSampler

class DataUtil():
    
    def __init__(self,sentence_len, voca_num, batch_size,dataset_name,val_path,voca_path):
    
        self.total = []
        self.word2id = {"<PAD>" : 0}
        self.sentence_len = sentence_len
        self.voca_num = voca_num
        self.batch_size = batch_size
        self.total_len = 0
        self.ps = PorterStemmer()
        self.stop_words = stopwords.words('english')
        self.val_path = val_path
        self.voca_path = voca_path
        
        self.dataset = json.load(open(dataset_name))
        self.valset = json.load(open(val_path))
        self.valX = []
        self.valy = []
        self.make_voca()
        
        self.make_total()
        self.num_batch = math.ceil(self.total_len/self.batch_size)
        
        self.set_val()

    def load_dict(self,dict_path):
        
        try:
            dict_file = json.load(open(dict_path))
            return dict_file
        except:
            return None
        
    def make_voca(self):
        
        mydict = self.load_dict(self.voca_path)

        if mydict is None :
            
            print("making voca...")
            voca_dict = {}
            for k in self.dataset.keys() :
                for sen in self.dataset[k] :
                    tok = self.sen2tok(sen)
                    for word in tok :
                        if word not in voca_dict:
                            voca_dict[word] = 1
                        else: voca_dict[word] += 1
        
            print("generated voca length is (before cutted) " , len(voca_dict))
        
            #sort can be better?
            for p in range(0,self.voca_num - 1) : 
                word = max(voca_dict, key=voca_dict.get)
                del voca_dict[word]
                self.word2id[word] = p + 1
                
            print("made voca")
            with open(self.voca_path, 'w') as outfiles:
                json.dump(self.word2id,outfiles)
        else : self.word2id = mydict
                
            
    def has_Number(self,s):
        return any(i.isdigit() for i in s)
            
    def sen2tok(self,string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " have", string) # \'ve
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " are", string) # \'re
        string = re.sub(r"\'d", " would", string) # \'d
        string = re.sub(r"\'ll", " will", string) # \'ll
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string) 
        
        string = [word for word in re.split("\W+", string)]
        
        ss = []
        for i in range(min(self.sentence_len,len(string))) :
            ss.append(string[i].lower())
        
        stopped_text = []
        for word in ss:
            wordlen = len(word)
            if wordlen < 2 or wordlen > 20 : continue
            # get rid of any string containing digits together in the process of filtering stopwords
            if word not in self.stop_words and (self.has_Number(word) == False):
                stopped_text.append(word)
            
        stemmed = [self.ps.stem(w) for w in stopped_text]
        
        return stemmed
    
    def tok2id(self,tok):
        res = []
        toklen = len(tok)
        for i in range(self.sentence_len):
            if i>=toklen : word = "<PAD>"
            else : word = tok[i]
            
            if word in self.word2id:
                res.append(self.word2id[word])
            else: res.append(self.word2id["<PAD>"])
            
            
        return res
    
    def sen2id(self,string):
        return self.tok2id(self.sen2tok(string))
    
    def make_total(self):
        
        article_polit = self.dataset['politics']
        article_enter = self.dataset['entertainment']
        article_sport = self.dataset['sport']
        article_buss = self.dataset['business']
        
        for i in range(len(article_polit)) :
            self.total.append([self.sen2id(self.dataset['politics'][i]),0])
        for i in range(len(article_enter)) :
            self.total.append([self.sen2id(self.dataset['entertainment'][i]),1])
        for i in range(len(article_sport)) :
            self.total.append([self.sen2id(self.dataset['sport'][i]),2])  
        for i in range(len(article_buss)) :
            self.total.append([self.sen2id(self.dataset['business'][i]),3])
            
        self.total_len = len(self.total)
        
    def testset_process(self,filename):
        
        dataset = json.load(open(filename))

        X = []
        y = []
        article_polit = dataset['politics']
        article_enter = dataset['entertainment']
        article_sport = dataset['sport']
        article_buss = dataset['business']
        
        for i in range(len(article_polit)) :
            X.append(self.sen2id(dataset['politics'][i]))
            y.append(0)
        for i in range(len(article_enter)) :
            X.append(self.sen2id(dataset['entertainment'][i]))
            y.append(1)
        for i in range(len(article_sport)) :
            X.append(self.sen2id(dataset['sport'][i]))
            y.append(2)
        for i in range(len(article_buss)) :
            X.append(self.sen2id(dataset['business'][i]))
            y.append(3)
        
        X = list(BatchSampler(X,batch_size = self.batch_size,drop_last = False))
        y = list(BatchSampler(y,batch_size = self.batch_size,drop_last = False))

        
        return X,y
        
    def make_batch(self):        

        art = []
        cat = []
        X = []
        y = []

        shuffle(self.total)
        
        for i in range(self.total_len):
            art.append(self.total[i][0])
            cat.append(self.total[i][1]) 
            
        m=0        
        for i in range(self.num_batch):
            batch_art=[]
            batch_cat=[]
            for j in range(self.batch_size):
                if m == self.total_len:
                    break
                batch_art.append(art[m])
                batch_cat.append(cat[m])
                m=m+1
            X.append(batch_art)
            y.append(batch_cat)
            
        return X,y
    
    def set_val(self):
        
        article_polit = self.valset['politics']
        article_enter = self.valset['entertainment']
        article_sport = self.valset['sport']
        article_buss = self.valset['business']
        
        for i in range(len(article_polit)) :
            self.valX.append(self.sen2id(self.valset['politics'][i]))
            self.valy.append(0)
        for i in range(len(article_enter)) :
            self.valX.append(self.sen2id(self.valset['entertainment'][i]))
            self.valy.append(1)
        for i in range(len(article_sport)) :
            self.valX.append(self.sen2id(self.valset['sport'][i]))
            self.valy.append(2)
        for i in range(len(article_buss)) :
            self.valX.append(self.sen2id(self.valset['business'][i]))
            self.valy.append(3)
            
    
    def make_val(self,size):
        
        X = self.valX
        y = self.valy
        
        p = list(zip(X,y))
        shuffle(p)
        p = list(zip(*p))
        X = list(p[0])
        y = list(p[1])
        
        X = X[:size]
        y = y[:size]
        
        X = list(BatchSampler(X,batch_size = self.batch_size,drop_last = False))
        y = list(BatchSampler(y,batch_size = self.batch_size,drop_last = False))
        
        return X,y
