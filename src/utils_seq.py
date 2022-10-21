from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import torch

from src.dataset import PfamDataset

def transform(seq):
  return " ".join(seq)

def transform_X(X):
    return [transform(elt) for elt in X] 

def transform_to_dataset(tokenizer, text, names, y, max_sequence_length):
    sequences = tokenizer.texts_to_sequences(text)
    query = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    query_tensor = torch.from_numpy(query)
    return PfamDataset(query_tensor, names, y)

def create_dataset(data, max_nb_chars=21,max_sequence_length=2975):
    X_train, names_train, y_train = data['train']['sequence'].values,data['train']['sequence_name'].values,data['train']['label'].values
    X_val, names_val, y_val = data['val']['sequence'].values, data['val']['sequence_name'].values,data['val']['label'].values
    X_test,names_test, y_test = data['test']['sequence'].values, data['test']['sequence_name'].values,data['test']['label'].values
    text_train = transform_X(X_train)
    text_val = transform_X(X_val)
    text_test = transform_X(X_test)
    train_tokenizer = Tokenizer(num_words=max_nb_chars)
    train_tokenizer.fit_on_texts(text_train)
    query_train = transform_to_dataset(tokenizer=train_tokenizer,
                                       text = text_train,names=names_train, 
                                       y = y_train, max_sequence_length=max_sequence_length)
    query_val = transform_to_dataset(tokenizer=train_tokenizer, 
                                     text = text_val, y = y_val,
                                     names=names_val, max_sequence_length=max_sequence_length)
    query_test = transform_to_dataset(tokenizer=train_tokenizer, 
                                      text = text_test, y = y_test,
                                      names=names_test, max_sequence_length=max_sequence_length)
    return query_train, query_val, query_test