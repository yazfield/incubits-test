import re
import argparse
from os import path

import pandas as pd
import numpy as np
from scipy.sparse import save_npz, load_npz
import gensim
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

def get_params():
    parser = argparse.ArgumentParser(description='Vectorizing documents')

    parser.add_argument('--vectorizer', type=str, default='count')
    parser.add_argument('--filename', type=str, default='data/vectors/data_vectors')
    parser.add_argument("--force", type=lambda x: bool(int(x)), default=True)
    parser.add_argument("--clean", type=lambda x: bool(int(x)), default=True)

    parser.add_argument('--max_df', default=0.8, type=float)
    parser.add_argument('--stop_words', default='english', type=str)
    parser.add_argument('--min_df', default=0.01, type=float)
    parser.add_argument('--max_features', default=90000, type=int)
    parser.add_argument("--binary", type=lambda x: bool(int(x)), default=False)
    parser.add_argument("--stem", type=lambda x: bool(int(x)), default=True)
    parser.add_argument('--ngram_from', default=1, type=int)
    parser.add_argument('--ngram_to', default=1, type=int)

    parser.add_argument('--size', default=700, type=int)
    parser.add_argument('--window', default=25, type=int)
    parser.add_argument('--min_count', default=5, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--alpha', default=0.025, type=float)
    parser.add_argument('--min_alpha', default=0.025, type=float)

    params = parser.parse_args()
    params.ngram_range = (params.ngram_from, params.ngram_to)
    if params.stop_words == 'None':
        params.stop_words = None

    return params

def slice_args(params):
    count_params = ['max_df', 'stop_words', 'min_df', 'max_df', 'max_features', 'binary', 'ngram_range']
    doc2vec_params = ['size', 'window', 'min_count', 'workers', 'alpha', 'min_alpha']
    if params.vectorizer == 'count':
        return {k:params[k] for k in vars(params).items() if k in count_params}
    elif params.vectorizer == 'tfidf':
        return {k:v for k,v in vars(params).items() if k in count_params}
    elif params.vectorizer == 'doc2vec':
        return {k:v for k,v in vars(params).items() if k in doc2vec_params}
    else:
        raise Exception('Vectorizer {} not supported'.format(params.vectorizer))
    

class DocIterator(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(list(filter(None, doc.split())), self.labels_list[idx])

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

def get_count_vectors(X, filename, stem=True, force=False, **kwargs):
    if not force and path.exists(filename):
        return load_npz(filename)

    if (stem):
        vectorizer = StemmedCountVectorizer(tokenizer=nltk.word_tokenize, **kwargs)
    else:
        vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize, **kwargs)

    X_bow = vectorizer.fit_transform(X)
    save_npz(filename, X_bow)

    return X_bow

def get_tfidf_vectors(X, filename, **kwargs):
    counts = get_count_vectors(X, filename, **kwargs)
    return TfidfTransformer(**kwargs).fit_transform(counts) 

def clean(text):
    # TODO: optimize this
    n = []
    stopw = stopwords.words('english')
    for txt in text.strip().lower().split():
        txt = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', txt)
        txt = re.sub('[\W_]', '', txt)
        txt = re.sub(' [0-9]+ ', ' ', txt)
        txt = ' '.join(txt.split())
        txt = re.sub('^[0-9]+ ', ' ', txt)
        txt = ' '.join(txt.split())
        if re.match('^[0-9]+$', txt):
            continue
        if STOPWORDS and txt in stopw:
            continue
        if txt:
            n.append(txt)
    return ' '.join(n)

if __name__ == '__main__':
    params = get_params()
    STOPWORDS = params.stop_words is not None

    print('Parameters:')
    for attr, value in sorted(vars(params).items()):
        print('{} = {}'.format(attr.upper(), value))
    print('')

    print('Loading data')

    training_text = pd.read_csv('data/training_text', sep='\|\|', engine='python', skiprows=1, names=['ID', 'Text'])
    training_variants = pd.read_csv('data/training_variants')
    train = pd.merge(training_text, training_variants, on='ID')
    train = train[train['Text'].apply(len) > 50]

    # oversampling small classes for DOC2VEC
    if(params.vectorizer == 'doc2vec'):
        train = train.append([train[train['Class'].isin([8, 9, 3])]] * 10)
        train = train.append([train[train['Class'].isin([6, 5])]]*3)

    docs = train['Text'].values if not params.clean else train['Text'].apply(clean)
    kwargs = slice_args(params)

    print('Vectorizing {} documents'.format(len(docs)))

    if params.vectorizer == 'count':
        get_count_vectors(docs, params.filename, **kwargs)
    elif params.vectorizer == 'tfidf':
        get_tfidf_vectors(docs, params.filename, **kwargs)
    elif params.vectorizer == 'doc2vec':
        print('cleaning other data')
        tags = np.vstack((train['Gene'].apply(clean), train['Variation'].apply(clean), train['Class'].values)).T
        it = DocIterator(docs, tags)
        print('Building vocab')
        model = Doc2Vec(**kwargs) # use fixed learning rate
        model.build_vocab(it)
        print('Training Doc2vec')
        model.train(it, total_examples=len(docs), epochs=10)
        print('Saving model')
        model.save(params.filename)
    else:
        raise Exception('Vectorizer {} not supported'.format(params.vectorizer))