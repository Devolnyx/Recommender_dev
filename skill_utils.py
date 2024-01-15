import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import re
import multiprocessing as mp
from multiprocessing import Pool

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

from scipy.sparse import csr_matrix, vstack
from sklearn.preprocessing import normalize

import spacy
from spacy.matcher import PhraseMatcher

from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor

# init params of skill extractor
nlp = spacy.load("en_core_web_lg")
# init skill extractor
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

# Download and unzip wordnet
try:
    nltk.data.find('wordnet.zip')
    nltk.data.find('stopwords.zip')
    nltk.data.find('punkt.zip')
except:
    nltk.download('punkt', download_dir='/nltk/')
    nltk.download('stopwords', download_dir='/nltk/')
    nltk.download('wordnet', download_dir='/nltk/')

    command = "unzip /nltk/punkt.zip -d /nltk/corpora"
    command = "unzip /nltk/stopwords.zip -d /nltk/corpora"
    command = "unzip /nltk/wordnet.zip -d /nltk/corpora"

    # subprocess.run(command.split())
    nltk.data.path.append('/nltk/')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

all_skills = dict((w, i) for (i, w) in enumerate(list(SKILL_DB.keys())))

def preprocess_text(text):
    # remove special characters and digits
    text = re.sub(r'\W+', ' ', text.lower())
    # tokenize the text
    tokens = word_tokenize(text)
    # remove stop words and perform lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    # rejoin the tokens into a string
    clean_text = ' '.join(tokens)
    return clean_text

def skills_embed(skills, dictionary=all_skills):

    idxss = [dictionary[word] for word in skills]
    embedding = np.zeros((1, len(dictionary)), dtype=np.uint8)

    for i, idxs in enumerate(idxss):
        embedding[0, idxs] = 1

    return normalize(csr_matrix(embedding), norm='l2', axis=1)

def extract_skills(record: str):
    annotations = skill_extractor.annotate(record)

    person_skills = [s['skill_id'] for s in annotations['results']['full_matches']]

    return person_skills

def parallel(function, arguments, n_threads=mp.cpu_count()//2):
    outputs = []
    with Pool(n_threads) as p:
        outputs = list(tqdm(p.imap(function, arguments), total=len(arguments)))

    return outputs

def dataset_skills_embed(df, column='clean_description'):
    embeddings = []

    for i in trange(len(df), ncols=80, desc='Total'):
        person_skills = extract_skills(df.iloc[i][column])
        embed = skills_embed(person_skills)
        embeddings.append(embed)

    return vstack(embeddings)

def dataset_skills_embed_parallel(df, column='clean_description'):
    embeddings = []

    persons_skills = parallel(function=extract_skills, arguments=df[column].to_list())
    skill_names = [', '.join([SKILL_DB[ID]['skill_name'] for ID in skills]) for skills in persons_skills]

    for i in trange(len(persons_skills), ncols=80, desc='Total'):
        embed = skills_embed(persons_skills[i])
        embeddings.append(embed)

    df['parser_skills'] = skill_names

    return vstack(embeddings), df
