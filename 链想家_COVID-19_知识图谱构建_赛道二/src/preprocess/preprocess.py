# -*- coding: utf-8 -*-
from collections import namedtuple
import json
import logging
logging.basicConfig(level=logging.ERROR,#控制台打印的日志级别
                    filename='preprocess.log',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s: %(message)s' ) #日志格式
                    
from transformers import AutoTokenizer 



TokenStruc = namedtuple('TokenStruc', ['token', 'start', 'end'])
EntityStruc = namedtuple('EntityStruc', ['word', 'tokens', 'start', 'end'])


class Sample:
    def __init__(self, text, spos):
        self.text = text
        self.tokenstruc_s = []
        self.spos = spos 
        self.spos_processed = []

        self.head_entitystruc_s = []


class SampleConvertor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _convert_sample(self, sample, idx):
        text = sample.text 
        spos = sample.spos 
        # assert spos != []
        tokenstruc_s = self._convert_text_to_tokenstruc_s(text)
        sample.tokenstruc_s = tokenstruc_s
        cnt = 0
        
        for spo in spos:
            try:
                head_entitystruc = self._convert_spo_one(tokenstruc_s, spo) # TODO
                sample.head_entitystruc_s.append(head_entitystruc)
            except ValueError as e:
                cnt = 1
                logging.error(f"error at train_{idx+1}, {str(e)}.")
                continue
        return  cnt

    def convert_sample_s(self, texts, spos_s):
        ccnt = 0
        samples = []
        for idx, text in enumerate(texts):
            text = text[f"train_{idx+1}"]
            spos = spos_s[idx][f"train_{idx+1}"]
            sample = Sample(text, spos)
            cnt = self._convert_sample(sample, idx)
            ccnt += cnt
            samples.append(sample)
            
        return ccnt
    
    def _convert_text_to_tokenstruc_s(self, text):
        def _token_start_end_aux(token, start, end, last_token):
            if token.startswith("##"):
                token = token.replace("##", "")
                new_start = end
                
            elif token in [",", ".", ":", ")","'","-", "/", "+", u"\u2014","%", "?", "=", "]", "_", u"\u2013"]:
                new_start = end 
                
            elif last_token in ["[","(", "'", "-", "/", u"\u2014", "%", "=", "_", "~", u"\u2013"]:
                new_start = end 

            else:
                new_start = end + 1  # "填入了一个空格"

            new_end = new_start + len(token)
            return new_start, new_end

        token_s = self.tokenizer.tokenize(text)
        tokenstruc_s = []

        start = 0
        end = len(token_s[0])
        tokenstruc_s.append(TokenStruc(token_s[0], start, end))

        for idx in range(len(token_s)-1):
            last_token = token_s[idx]
            token = token_s[idx+1]
            start, end = _token_start_end_aux(token, start, end, last_token)
            tokenstruc_s.append( TokenStruc(token, start, end) )

        return tokenstruc_s 

    def _convert_label_s(self, label_s):
        pass 
    
    def _convert_spo_one(self, tokenstruc_s, spo):
        head = spo["head"]
        tail = spo["tail"]
        rel = spo["rel"] 

        head_entitystruc = self._find_entity_in_tokenstruc_s(tokenstruc_s, head)
        return head_entitystruc

    def _find_entity_in_tokenstruc_s(self, tokenstruc_s, entity):
        entity_start = entity["start"]
        entity_end = entity["end"]
        entity_word = entity["word"]

        for idx, tokenstruc in enumerate(tokenstruc_s):
            if entity_start == tokenstruc.start:
                entity_start_in_tokenstruc_s = idx
                break
        else:
            entity_start_in_tokenstruc_s = None

        for idx, tokenstruc in enumerate(tokenstruc_s):
            if entity_end == tokenstruc.end:
                entity_end_in_tokenstruc_s = idx
                break
        else:
            entity_end_in_tokenstruc_s = None

        if entity_start_in_tokenstruc_s is not None and entity_end_in_tokenstruc_s is not None:
            return EntityStruc(entity_word, self.tokenizer.tokenize(entity_word),
                                 entity_start_in_tokenstruc_s, entity_end_in_tokenstruc_s)
        else:
            raise ValueError(f"can't find entity {entity_word} {entity_start} {entity_end} index in tokenstruc_s")


if __name__ == "__main__":
    # tst of tokenizer
    pretrain_weight = r"D:\data\预训练模型\bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(pretrain_weight)
    test_sampleConvertor = SampleConvertor(tokenizer)

    
    with open("./链想家_COVID-19_知识图谱构建_赛道二/data/task2_public/task2_user_train.json", "r", encoding="utf-8") as f:
        train_texts = json.load(f)
        print(len(train_texts))

    with open("./链想家_COVID-19_知识图谱构建_赛道二/data/task2_public/task2_train_label.json", "r", encoding="utf-8") as f:
        train_labels = json.load(f)
    
    # all test ========================================
    ccnt = test_sampleConvertor.convert_sample_s(train_texts, train_labels)
    print(ccnt)
  
    # single test ========================
    # idx = 23
    # test_sentence = train_texts[idx][f"train_{idx+1}"]
    # rst_ts_s = test_sampleConvertor._convert_text_to_tokenstruc_s(test_sentence)  
    # for i in rst_ts_s:
    #     print(i)  

    # test_labels = train_labels[idx][f"train_{idx+1}"]
    # for test_label in test_labels:
    #     es = test_sampleConvertor._convert_spo_one(rst_ts_s, test_label)


    
    # print(len(rst))
    pass 