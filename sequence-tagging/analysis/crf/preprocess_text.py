import re
import spacy
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-num")
spacy_tokenizer = spacy.load("en_core_web_sm")

sentence = "Total net sales decreased 2% or $5.4 billion during 2019 compared to 2018."

def sec_bert_num_preprocess(text):
    tokens = [t.text for t in spacy_tokenizer(text)]

    processed_text = []
    for token in tokens:
        if re.fullmatch(r"(\d+[\d,.]*)|([,.]\d+)", token):
            processed_text.append('[NUM]')
        else:
            processed_text.append(token)
            
    return ' '.join(processed_text)
        
tokenized_sentence = tokenizer.tokenize(sec_bert_num_preprocess(sentence))
print(tokenized_sentence)
"""
['total', 'net', 'sales', 'decreased', '[NUM]', '%', 'or', '$', '[NUM]', 'billion', 'during', '[NUM]', 'compared', 'to', '[NUM]', '.']
"""