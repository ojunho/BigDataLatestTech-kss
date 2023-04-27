from gensim.models import Word2Vec
from konlpy.tag import Kkma
from konlpy.tag import Okt
import re

def split_sentences(text):
    # Define regex pattern to split Korean text into sentences
    pattern = re.compile(".*?[.?!]")

    # Split the text into sentences using the regex pattern
    sentences = []
    for line in text.split('\n'):
        for match in pattern.findall(line):
            if match.strip():
                sentences.append(match.strip())

    return sentences

# Load the Korean text file
with open('ko_wiki_text_utf8.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenize the text using Okt
okt = Okt()
sentences = split_sentences(text)

# Train a Word2Vec model on the tokenized sentences
model = Word2Vec.load('word2vec-ko_wiki_KMA_utf8.model')
#model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)

# Compute the similarity between the input sentence and each sentence in the text file
input_sentence = "나는 자랑스러운 대한민국의 육군 전사다."
input_tokens = okt.nouns(input_sentence)  # tokenize the input sentence using Okt
similar_sentences = []
for sentence in sentences:
    sentence_tokens = okt.nouns(sentence)
    if len(sentence_tokens) == 0 or len(input_tokens) == 0:
        continue
    similarity = model.wv.n_similarity(input_tokens, sentence_tokens)
    similar_sentences.append((sentence, similarity))

# Sort the similar sentences by similarity score in descending order and output the top n sentences
n = 5
top_similar_sentences = sorted(similar_sentences, key=lambda x: x[1], reverse=True)[:n]
for sentence, similarity in top_similar_sentences:
    print("\nsentence: ", sentence, "\nsimilarity: ", similarity)
