# 1. Import libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
# --------------------------------------------------------------------------------------
# 2. Download resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
# Download the missing 'punkt_tab' resource
nltk.download('punkt_tab') # This line is added to download the 'punkt_tab' resource
# --------------------------------------------------------------------------------------
# 3. Write text
text = ("Hello everyone! I am first name last name. "
        "I am a loyal KSKA Git user all the way from Sangamwadi Empire. "
        "I have considerable knowledge about life, Python, C++, Java, Rust, Golang and Blockchain. "
        "For every smart contract, I lose one strand of my hair. "
        "In my free time, which by the way, I barely get, I like to swim.")
# --------------------------------------------------------------------------------------
# 4. Sentence tokenization
sentences = sent_tokenize(text)
print("Sentence Tokenization:\n", sentences)
print("==============================================================")

# 5. Word tokenization
words = word_tokenize(text)
print("Word Tokenization:\n", words)
print("==============================================================")

# 6. Removing punctuation and stop words
stop_words = set(stopwords.words('english'))
text_alpha = re.sub('[^a-zA-Z]', ' ', text)  # Keep only letters
print("Text after removing non-letters:\n", text_alpha)
print("==============================================================")

# Tokenize again after cleaning
tokens_clean = word_tokenize(text_alpha.lower())
filtered_words = [word for word in tokens_clean if word not in stop_words]
print("Filtered Tokens (after removing stopwords):\n", filtered_words)
print("==============================================================")

# 7. Stemming
words_to_stem = ["write", "writing", "wrote", "writes", "reading", "reads"]
ps = PorterStemmer()
print("Stemming Examples:")
for w in words_to_stem:
    print(f"{w} --> {ps.stem(w)}")
print("==============================================================")

# 8. Lemmatization
lemmatizer = WordNetLemmatizer()
text_lemmatize = "studies studying cries cry"
tokens_lem = nltk.word_tokenize(text_lemmatize)
print("Lemmatization Examples:")
for w in tokens_lem:
    print(f"{w} --> {lemmatizer.lemmatize(w)}")