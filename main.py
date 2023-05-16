import nltk
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stopwords]
    return tokens

def calculate_word_frequencies(tokens):
    word_freq = Counter(tokens)
    return word_freq

def visualize_word_frequencies(word_freq, top_n = 15):
    top_words = word_freq.most_common(top_n)
    words, frequencies = zip(*top_words)

    plt.bar(words, frequencies)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f'Top {top_n} Words by Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def generate_word_cloud(word_freq):
    wordcloud = WordCloud(width = 800, height = 400).generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    with open('document.txt', 'r') as file:
        text = file.read()

    tokens = preprocess_text(text)
    word_freq = calculate_word_frequencies(tokens)

    visualize_word_frequencies(word_freq, top_n = 15)
    generate_word_cloud(word_freq)
