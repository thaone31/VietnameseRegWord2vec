import re,os                                   # 're' Replication of text.
import numpy as np
import pandas as pd                         # 'pandas' to manipulate the dataset.
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential                # 'Sequential' model will be used for training.
from sklearn.model_selection import train_test_split          # 'train_test_split' for splitting the data into train and test data.
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences       # 'pad_sequences' for having same dimmension for each sequence.
from tensorflow.keras.layers import Embedding, Bidirectional,LSTM, Flatten, Dense,Input,Average,Reshape,Dropout,Concatenate,Maximum     # import some layers for training.
from tensorflow.keras.layers import Conv2D, MaxPool2D,Convolution1D,MaxPooling1D     # import some layers for training.
from tensorflow.keras.utils import to_categorical
import gensim.models.keyedvectors as word2vec #need to use due to depreceated model
# import gensim.models.word2vec as Word2Vec #need to use due to depreceated model
import gc
from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from tensorflow.keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

import pandas as pd
from keras.utils import to_categorical

label_mapping = {'pos': 1, 'neu': 2, 'neg': 3}

X_train = pd.read_excel(r"./VietnameseRegWord2vec/Hotel_sentiment-20240207T135933Z-002/Hotel_sentiment/data/Data_train.xlsx")

if X_train.isnull().values.any():
    X_train = X_train.dropna()

print(X_train.shape)

X_train = X_train[['processed_title', 'processed_review', 'user_rate']]
X_train_title = X_train['processed_title'].apply(str)
X_train_text = X_train['processed_review'].apply(str)

# Map string labels to integer values
y_train = X_train['user_rate'].map(label_mapping)

# Convert to one-hot encoding
train_labels = to_categorical(y_train - 1, num_classes=3)
import pandas as pd
from keras.utils import to_categorical

# Assuming 'pos', 'neu', 'neg' are your unique classes
label_mapping = {'pos': 1, 'neu': 2, 'neg': 3}

X_test = pd.read_excel(r"./VietnameseRegWord2vec/Hotel_sentiment-20240207T135933Z-002/Hotel_sentiment/data/Data_test.xlsx")

if X_test.isnull().values.any():
    X_test = X_test.dropna()

print(X_test.shape)
print(X_test.columns)

X_test = X_test[['processed_title', 'processed_review', 'user_rate']]
X_test_title = X_test['processed_title'].apply(str)
X_test_text = X_test['processed_review'].apply(str)

# Map string labels to integer values
y_test = X_test['user_rate'].map(label_mapping)

# Convert to one-hot encoding
test_labels = to_categorical(y_test - 1, num_classes=3)


print(X_train['user_rate'].unique())

from gensim.models import KeyedVectors

w2vModel = word2vec.KeyedVectors.load_word2vec_format('./VietnameseRegWord2vec/Hotel_sentiment-20240207T135933Z-002/Hotel_sentiment/baomoi.model.bin', binary=True, limit=50000)

tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train_title)
tokenizer.fit_on_texts(X_train_text)
tokenizer.fit_on_texts(X_test_title)
tokenizer.fit_on_texts(X_test_text)

train_title = tokenizer.texts_to_sequences(X_train_title)
train_text = tokenizer.texts_to_sequences(X_train_text)
test_title = tokenizer.texts_to_sequences(X_test_title)
test_text = tokenizer.texts_to_sequences(X_test_text)

from sklearn.model_selection import train_test_split

train_sent_titles, val_sent_titles, train_sent_texts, val_sent_texts, train_ratings, val_ratings = train_test_split(train_title, train_text, train_labels, test_size=0.1)
print(len(train_sent_texts), len(val_sent_texts))
print(len(train_sent_titles), len(val_sent_titles))
print(len(train_ratings), len(val_ratings))

MAX_LEN = 256

def convert_sents_ids(sents):
    ids = []
    for sent in sents:
        sent = str(sent)
        # Split the sentence into words and encode them individually
        encoded_sent = [w2vModel[word] if word in w2vModel else np.zeros(w2vModel.vector_size) for word in sent.split()]
        ids.append(encoded_sent)

    # Pad sequences to a fixed length using Keras
    ids = pad_sequences(ids, maxlen=MAX_LEN, dtype="float32", value=0, truncating="post", padding="post")
    return ids


train_title_ids = convert_sents_ids(train_sent_titles)
train_text_ids = convert_sents_ids(train_sent_texts)
val_title_ids = convert_sents_ids(val_sent_titles)
val_text_ids = convert_sents_ids(val_sent_texts)
test_title_ids = convert_sents_ids(test_title)
test_text_ids = convert_sents_ids(test_text)

def make_mask(batch_ids):
    batch_mask = []
    for ids in batch_ids:
        mask = [int(token_id > 0) for token_id in ids[0]]
        batch_mask.append(mask)
    return np.array(batch_mask)

train_title_masks = make_mask(train_title_ids)
train_text_masks = make_mask(train_text_ids)

val_title_masks = make_mask(val_title_ids)
val_text_masks = make_mask(val_text_ids)

test_title_masks = make_mask(test_title_ids)
test_text_masks = make_mask(test_text_ids)

import numpy as np

def make_data_loader(ids, masks, labels, BATCH_SIZE=4):
    # Check consistency of sample lengths
    if len(ids) != len(masks) or len(ids) != len(labels):
        raise ValueError("Inconsistent number of samples in input variables")

    print("Lengths: ids={}, masks={}, labels={}".format(len(ids), len(masks), len(labels)))

    data = list(zip(ids, masks, labels))
    np.random.shuffle(data)  # Shuffle the data

    batched_data = [data[i:i + BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]

    return batched_data


train_title_dataloader = make_data_loader(train_title_ids, train_title_masks, train_ratings)
train_text_dataloader = make_data_loader(train_text_ids, train_text_masks, train_ratings)

val_title_dataloader = make_data_loader(val_title_ids, val_title_masks, val_ratings)
val_text_dataloader = make_data_loader(val_text_ids, val_text_masks, val_ratings)

test_title_dataloader = make_data_loader(test_title_ids, test_title_masks, test_labels)
test_text_dataloader = make_data_loader(test_text_ids, test_text_masks, test_labels)


import numpy as np

# # Assuming train_title_ids, train_title_masks, and train_labels are your training data
# num_samples_to_keep = len(test_title_ids)  # Use the length of your testing dataset

# # Randomly sample a subset of the training data
# random_indices = np.random.choice(len(train_title_ids), num_samples_to_keep, replace=False)

# # Update your training data with the randomly sampled subset
# train_title_ids = train_title_ids[random_indices]
# train_title_masks = train_title_masks[random_indices]

# train_text_ids = train_text_ids[random_indices]
# train_text_masks = train_text_masks[random_indices]

# train_labels = train_labels[random_indices]

import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assuming train_labels and test_labels are NumPy arrays
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

y_train_1d = np.argmax(y_train, axis=1)

lr = LogisticRegression()

train_title_ids_2d = np.array(train_title_ids).reshape(len(train_title_ids), -1)
train_text_ids_2d = np.array(train_text_ids).reshape(len(train_text_ids), -1)

train_data = np.concatenate((train_title_ids_2d, train_text_ids_2d), axis=1)

lr.fit(train_data, y_train_1d)

test_title_ids_2d = np.array(test_title_ids).reshape(len(test_title_ids), -1)
test_text_ids_2d = np.array(test_text_ids).reshape(len(test_text_ids), -1)

test_data = np.concatenate((test_title_ids_2d, test_text_ids_2d), axis=1)

y_pred_lr = lr.predict(test_data)

acc_lr = accuracy_score(np.argmax(y_test, axis=1), y_pred_lr)
conf = confusion_matrix(np.argmax(y_test, axis=1), y_pred_lr)
clf_report = classification_report(np.argmax(y_test, axis=1), y_pred_lr)

print(f"Accuracy Score of Logistic Regression is: {acc_lr}")
print(f"Confusion Matrix:\n{conf}")
print(f"Classification Report:\n{clf_report}")



