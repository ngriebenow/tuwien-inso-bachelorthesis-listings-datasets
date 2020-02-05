import torch
from transformers import *

MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
		  (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
		  (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
		  (RobertaModel,    RobertaTokenizer,    'roberta-base'),
		  (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base')]


for model_class, tokenizer_class, pretrained_weights in MODELS:
	tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
	model = model_class.from_pretrained(pretrained_weights)

BERT_MODEL_CLASSES = [BertModel, BertForNextSentencePrediction,
BertForSequenceClassification, BertForQuestionAnswering]

for model_class in BERT_MODEL_CLASSES:
	model = model_class.from_pretrained('bert-base-uncased')
