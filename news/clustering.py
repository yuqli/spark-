import glob
import string
import re

alt_train_path = './data/alt.atheism-train/*'
alt_test_path = './data/alt.atheism-test/*'
comp_train_path = './data/comp.graphics/*'
comp_test_path = './data/comp.graphics-train/*'

alt_train = []

alt_train_files = glob.glob(alt_train_path, recursive=True)
alt_train_files

for fle in alt_train_files:
   # open the file and then call .read() to get the text
   with open(fle, 'rb') as f:
      text = f.read()
      alt_train.append(text)

len(alt_train)

def preprocess(text):
    '''
    This function takes a string of text and does the following:
    - Remove initial "*" characters -- these are useless and extra noise
    - Convert alphabetic characters to lower case (Hello --> hello)
    - Replace numeric characters with "#" character.
    '''
    text = text.replace('*','').lower()
    text = re.sub('\d', '*', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('', '', '1234567890'))
    return text

def byToString(textlist):
    for i in range(len(textlist)):
        try:
            textlist[i] = textlist[i].decode('utf-8')
        except UnicodeDecodeError:
            textlist[i] = None # discard the files cannot be decoded
    textlist = [x for x in textlist if x is not None]
    return textlist

alt_train = byToString(alt_train)

type(alt_train[0])

alt_train = [preprocess(item) for item in alt_train]
alt_train[0]

# https://github.com/kundan-git/pyspark-text-classification/blob/master/Spark-Text-Classification.ipynb
