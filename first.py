import nltk
import ssl
import pandas as pd
# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')



# import pandas as pd
import csv

# Sample social media posts with various text challenges
social_media_data = [
    {"id": 1, "text": "OMG!!! Just had the BEST pizza ever 🍕🍕🍕 #foodie #yummy", "platform": "twitter"},
    {"id": 2, "text": "Can't believe it's Monday again... 😴 Need more coffee!!!", "platform": "facebook"},
    {"id": 3, "text": "Working on my new project. It's challenging but exciting! #coding #python", "platform": "linkedin"},
    {"id": 4, "text": "The weather is absolutely TERRIBLE today!!! Rain, rain, rain... ☔", "platform": "twitter"},
    {"id": 5, "text": "Just finished reading an amazing book. Highly recommend it to everyone!", "platform": "facebook"},
    {"id": 6, "text": "Meeting with clients tomorrow. Hope everything goes well! #business", "platform": "linkedin"},
    {"id": 7, "text": "LOL that movie was hilarious 😂😂😂 Can't stop laughing!", "platform": "twitter"},
    {"id": 8, "text": "Studying for exams is so stressful... But I know I can do it! 💪", "platform": "facebook"},
    {"id": 9, "text": "Beautiful sunset today! Nature is truly amazing", "platform":"facebook"},
    {"id": 10, "text": "Traffic is CRAZY today!!! Why does everyone drive so slowly???", "platform": "twitter"}
]

# Create DataFrame and save to CSV
df = pd.DataFrame(social_media_data)
df.to_csv('social_media_posts.csv', index=False)
print("Sample social media dataset created successfully!")
print(f"Dataset contains {len(df)} posts")

import pandas as pd
import re

# Load the dataset
df = pd.read_csv('social_media_posts.csv')

def convert_to_lowercase(text):
    """Convert text to lowercase"""
    return text.lower()

# Apply lowercasing
df['text_lowercase'] = df['text'].apply(convert_to_lowercase)

print("=== LOWERCASING RESULTS ===")
print("Original vs Lowercase:")
for i in range(3):
    print(f"Original: {df['text'].iloc[i]}")
    print(f"Lowercase: {df['text_lowercase'].iloc[i]}")
    print("-" * 50)


import string

def remove_punctuation(text):
    """Remove punctuation and special characters"""
    # Remove standard punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def remove_emojis_and_special(text):
    """Remove emojis and special characters"""
    # Remove emojis and special unicode characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Apply punctuation removal
df['text_no_punct'] = df['text_lowercase'].apply(remove_punctuation)
df['text_clean'] = df['text_no_punct'].apply(remove_emojis_and_special)

print("=== PUNCTUATION REMOVAL RESULTS ===")
for i in range(3):
    print(f"Original: {df['text'].iloc[i]}")
    print(f"After cleaning: {df['text_clean'].iloc[i]}")
    print("-" * 50)



import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

def tokenize_words(text):
    """Tokenize text into words"""
    tokens = word_tokenize(text)
    return tokens

def tokenize_sentences(text):
    """Tokenize text into sentences"""
    sentences = sent_tokenize(text)
    return sentences

# Apply tokenization
df['word_tokens'] = df['text_clean'].apply(tokenize_words)

print("=== TOKENIZATION RESULTS ===")
for i in range(3):
    print(f"Original text: {df['text'].iloc[i]}")
    print(f"Word tokens: {df['word_tokens'].iloc[i]}")
    print(f"Number of tokens: {len(df['word_tokens'].iloc[i])}")
    print("-" * 50)

# Show tokenization statistics
total_tokens = sum(len(tokens) for tokens in df['word_tokens'])
avg_tokens = total_tokens / len(df)
print(f"\nTokenization Statistics:")
print(f"Total tokens across all posts: {total_tokens}")
print(f"Average tokens per post: {avg_tokens:.2f}")


def advanced_tokenize(text):
    """Advanced tokenization handling edge cases"""
    # Handle contractions and special cases
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove empty tokens
    tokens = [token for token in tokens if token.strip()]

    return tokens

# Apply advanced tokenization
df['advanced_tokens'] = df['text_clean'].apply(advanced_tokenize)

print("=== ADVANCED TOKENIZATION COMPARISON ===")
for i in range(2):
    print(f"Basic tokens: {df['word_tokens'].iloc[i]}")
    print(f"Advanced tokens: {df['advanced_tokens'].iloc[i]}")
    print("-" * 50)


from nltk.corpus import stopwords

# Get English stopwords
stop_words = set(stopwords.words('english'))

print("=== STOPWORDS INFORMATION ===")
print(f"Number of English stopwords: {len(stop_words)}")
print("Sample stopwords:", list(stop_words)[:20])

def remove_stopwords(tokens):
    """Remove stopwords from token list"""
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

# Apply stopword removal
df['tokens_no_stopwords'] = df['advanced_tokens'].apply(remove_stopwords)

print("\n=== STOPWORD REMOVAL RESULTS ===")
for i in range(3):
    print(f"Original tokens: {df['advanced_tokens'].iloc[i]}")
    print(f"After stopword removal: {df['tokens_no_stopwords'].iloc[i]}")
    print(f"Tokens removed: {len(df['advanced_tokens'].iloc[i]) - len(df['tokens_no_stopwords'].iloc[i])}")
    print("-" * 50)


from nltk.stem import PorterStemmer

# Initialize Porter Stemmer
porter_stemmer = PorterStemmer()

def apply_stemming(tokens):
    """Apply Porter stemming to tokens"""
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return stemmed_tokens

# Apply stemming
df['stemmed_tokens'] = df['tokens_no_stopwords'].apply(apply_stemming)

print("=== STEMMING RESULTS ===")
for i in range(3):
    print(f"Original tokens: {df['tokens_no_stopwords'].iloc[i]}")
    print(f"Stemmed tokens: {df['stemmed_tokens'].iloc[i]}")
    print("-" * 50)

# Show stemming examples
print("\n=== STEMMING EXAMPLES ===")
sample_words = ['running', 'runs', 'ran', 'easily', 'fairly', 'studies', 'studying', 'studied']
for word in sample_words:
    stemmed = porter_stemmer.stem(word)
    print(f"{word} -> {stemmed}")


from nltk.stem import SnowballStemmer

# Initialize Snowball Stemmer
snowball_stemmer = SnowballStemmer('english')

def apply_snowball_stemming(tokens):
    """Apply Snowball stemming to tokens"""
    stemmed_tokens = [snowball_stemmer.stem(token) for token in tokens]
    return stemmed_tokens

# Apply Snowball stemming
df['snowball_stemmed'] = df['tokens_no_stopwords'].apply(apply_snowball_stemming)

print("=== STEMMING ALGORITHM COMPARISON ===")
test_words = ['running', 'studies', 'beautiful', 'organization', 'generously']

for word in test_words:
    porter_result = porter_stemmer.stem(word)
    snowball_result = snowball_stemmer.stem(word)
    print(f"{word:12} | Porter: {porter_result:10} | Snowball: {snowball_result}")

print("\n=== STEMMING ON SAMPLE DATA ===")
for i in range(2):
    print(f"Original: {df['tokens_no_stopwords'].iloc[i]}")
    print(f"Porter: {df['stemmed_tokens'].iloc[i]}")
    print(f"Snowball: {df['snowball_stemmed'].iloc[i]}")
    print("-" * 50)


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatizer accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def apply_lemmatization(tokens):
    """Apply lemmatization with POS tagging"""
    lemmatized_tokens = []
    for token in tokens:
        pos = get_wordnet_pos(token)
        lemmatized_token = lemmatizer.lemmatize(token, pos)
        lemmatized_tokens.append(lemmatized_token)
    return lemmatized_tokens

def simple_lemmatization(tokens):
    """Apply simple lemmatization without POS tagging"""
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# Apply both types of lemmatization
df['simple_lemmatized'] = df['tokens_no_stopwords'].apply(simple_lemmatization)
df['pos_lemmatized'] = df['tokens_no_stopwords'].apply(apply_lemmatization)

print("=== LEMMATIZATION RESULTS ===")
for i in range(3):
    print(f"Original: {df['tokens_no_stopwords'].iloc[i]}")
    print(f"Simple lemmatized: {df['simple_lemmatized'].iloc[i]}")
    print(f"POS lemmatized: {df['pos_lemmatized'].iloc[i]}")
    print("-" * 50)


print("=== STEMMING VS LEMMATIZATION COMPARISON ===")

# Test words that show clear differences
test_words = ['running', 'better', 'studies', 'feet', 'geese', 'children', 'mice']

print(f"{'Word':<12} | {'Stemmed':<12} | {'Lemmatized':<12}")
print("-" * 40)

for word in test_words:
    stemmed = porter_stemmer.stem(word)
    lemmatized = lemmatizer.lemmatize(word, get_wordnet_pos(word))
    print(f"{word:<12} | {stemmed:<12} | {lemmatized:<12}")

print("\n=== COMPARISON ON SAMPLE DATA ===")
for i in range(2):
    print(f"Original tokens: {df['tokens_no_stopwords'].iloc[i]}")
    print(f"Stemmed: {df['stemmed_tokens'].iloc[i]}")
    print(f"Lemmatized: {df['pos_lemmatized'].iloc[i]}")
    print("-" * 50)

# Calculate processing statistics
def calculate_unique_tokens(token_lists):
    all_tokens = []
    for tokens in token_lists:
        all_tokens.extend(tokens)
    return len(set(all_tokens))

original_unique = calculate_unique_tokens(df['tokens_no_stopwords'])
stemmed_unique = calculate_unique_tokens(df['stemmed_tokens'])
lemmatized_unique = calculate_unique_tokens(df['pos_lemmatized'])

print(f"\n=== VOCABULARY REDUCTION STATISTICS ===")
print(f"Original unique tokens: {original_unique}")
print(f"After stemming: {stemmed_unique} ({((original_unique-stemmed_unique)/original_unique)*100:.1f}% reduction)")
print(f"After lemmatization: {lemmatized_unique} ({((original_unique-lemmatized_unique)/original_unique)*100:.1f}% reduction)")


import pandas as pd
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        # Add social media stopwords
        social_media_stopwords = {
            'lol', 'omg', 'wtf', 'tbh', 'imo', 'imho', 'btw', 'fyi',
            'rt', 'dm', 'pm', 'aka', 'asap', 'fomo', 'yolo', 'tbt'
        }
        self.extended_stop_words = self.stop_words.union(social_media_stopwords)

    def clean_text(self, text):
        """Apply basic text cleaning"""
        # Convert to lowercase
        text = text.lower()

        # Handle contractions
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'d", " would", text)

        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def tokenize(self, text):
        """Tokenize text into words"""
        tokens = word_tokenize(text)
        return [token for token in tokens if token.strip()]

    def remove_stopwords(self, tokens, extended=True):
        """Remove stopwords"""
        stop_set = self.extended_stop_words if extended else self.stop_words
        return [token for token in tokens if token.lower() not in stop_set]

    def get_wordnet_pos(self, word):
        """Get POS tag for lemmatization"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def stem_tokens(self, tokens):
        """Apply stemming"""
        return [self.stemmer.stem(token) for token in tokens]

    def lemmatize_tokens(self, tokens):
        """Apply lemmatization with POS tagging"""
        return [self.lemmatizer.lemmatize(token, self.get_wordnet_pos(token)) for token in tokens]

    def preprocess(self, text, method='lemmatize', remove_stopwords=True):
        """Complete preprocessing pipeline"""
        # Clean text
        cleaned_text = self.clean_text(text)

        # Tokenize
        tokens = self.tokenize(cleaned_text)

        # Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)

        # Apply stemming or lemmatization
        if method == 'stem':
            tokens = self.stem_tokens(tokens)
        elif method == 'lemmatize':
            tokens = self.lemmatize_tokens(tokens)

        return tokens

# Initialize preprocessor
preprocessor = TextPreprocessor()

# Load data
df = pd.read_csv('social_media_posts.csv')

# Apply complete preprocessing pipeline
df['processed_stem'] = df['text'].apply(lambda x: preprocessor.preprocess(x, method='stem'))
df['processed_lemma'] = df['text'].apply(lambda x: preprocessor.preprocess(x, method='lemmatize'))
df['processed_tokens_only'] = df['text'].apply(lambda x: preprocessor.preprocess(x, method='none'))

print("=== COMPLETE PREPROCESSING PIPELINE RESULTS ===")
for i in range(3):
    print(f"Original: {df['text'].iloc[i]}")
    print(f"Tokens only: {df['processed_tokens_only'].iloc[i]}")
    print(f"With stemming: {df['processed_stem'].iloc[i]}")
    print(f"With lemmatization: {df['processed_lemma'].iloc[i]}")
    print("=" * 60)


def analyze_preprocessing_results(df):
    """Analyze the results of different preprocessing methods"""
    
    print("=== PREPROCESSING ANALYSIS ===")
    
    # Calculate statistics for each method
    methods = {
        'Original': 'text',
        'Tokens Only': 'processed_tokens_only',
        'With Stemming': 'processed_stem',
        'With Lemmatization': 'processed_lemma'
    }
    
    stats = {}
    
    for method_name, column in methods.items():
        if column == 'text':
            # For original text, count words
            word_counts = [len(text.split()) for text in df[column]]
            unique_words = set()
            for text in df[column]:
                unique_words.update(text.lower().split())
        else:
            # For processed tokens
            word_counts = [len(tokens) for tokens in df[column]]
            unique_words = set()
            for tokens in df[column]:
                unique_words.update(tokens)
        
        stats[method_name] = {
            'avg_words': sum(word_counts) / len(word_counts),
            'total_words': sum(word_counts),
            'unique_words': len(unique_words)
        }
    
    # Display statistics
    print(f"{'Method':<20} | {'Avg Words':<10} | {'Total Words':<12} | {'Unique Words':<12}")
    print("-" * 65)
    
    for method, stat in stats.items():
        print(f"{method:<20} | {stat['avg_words']:<10.1f} | {stat['total_words']:<12} | {stat['unique_words']:<12}")
    
    # Calculate reduction percentages
    original_unique = stats['Original']['unique_words']
    stem_unique = stats['With Stemming']['unique_words']
    lemma_unique = stats['With Lemmatization']['unique_words']
    
    print(f"\n=== VOCABULARY REDUCTION ===")
    print(f"Stemming reduction: {((original_unique - stem_unique) / original_unique) * 100:.1f}%")
    print(f"Lemmatization reduction: {((original_unique - lemma_unique) / original_unique) * 100:.1f}%")

# Run analysis
analyze_preprocessing_results(df)

# Save processed data
df.to_csv('processed_social_media_posts.csv', index=False)
print("\nProcessed data saved to 'processed_social_media_posts.csv'")


import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

def plot_word_frequencies(tokens_list, title, top_n=10):
    """Plot word frequency distribution"""
    # Flatten all tokens
    all_tokens = []
    for tokens in tokens_list:
        all_tokens.extend(tokens)

    # Count frequencies
    word_freq = Counter(all_tokens)
    most_common = word_freq.most_common(top_n)

    # Create plot
    words, frequencies = zip(*most_common)

    plt.figure(figsize=(12, 6))
    plt.bar(words, frequencies)
    plt.title(f'Top {top_n} Words - {title}')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return word_freq

# Load processed data
df = pd.read_csv('processed_social_media_posts.csv')

# Convert string representations back to lists (if needed)
import ast

def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

df['processed_stem'] = df['processed_stem'].apply(safe_eval)
df['processed_lemma'] = df['processed_lemma'].apply(safe_eval)

# Create frequency plots
print("=== WORD FREQUENCY ANALYSIS ===")

# Original text frequency (simple split)
original_words = []
for text in df['text']:
    words = text.lower().split()
    original_words.extend([word.strip('.,!?";:()[]{}') for word in words])

original_freq = Counter(original_words)
print("Top 10 words in original text:")
for word, freq in original_freq.most_common(10):
    print(f"{word}: {freq}")

# Stemmed words frequency
stemmed_freq = plot_word_frequencies(df['processed_stem'], 'Stemmed Words')

# Lemmatized words frequency
lemmatized_freq = plot_word_frequencies(df['processed_lemma'], 'Lemmatized Words')

def compare_processing_methods(df):
    """Compare different text processing methods"""

    print("=== PROCESSING METHOD COMPARISON ===")

    # Sample text for detailed comparison
    sample_text = df['text'].iloc[0]
    print(f"Sample text: {sample_text}")
    print()

    # Show step-by-step processing
    preprocessor = TextPreprocessor()

    # Step 1: Clean text
    cleaned = preprocessor.clean_text(sample_text)
    print(f"1. Cleaned: {cleaned}")

    # Step 2: Tokenize
    tokens = preprocessor.tokenize(cleaned)
    print(f"2. Tokenized: {tokens}")

    # Step 3: Remove stopwords
    no_stopwords = preprocessor.remove_stopwords(tokens)
    print(f"3. No stopwords: {no_stopwords}")

    # Step 4a: Stem
    stemmed = preprocessor.stem_tokens(no_stopwords)
    print(f"4a. Stemmed: {stemmed}")

    # Step 4b: Lemmatize
    lemmatized = preprocessor.lemmatize_tokens(no_stopwords)
    print(f"4b. Lemmatized: {lemmatized}")

    print("\n" + "="*60)

    # Overall statistics
    print("=== OVERALL PROCESSING STATISTICS ===")

    total_original_chars = sum(len(text) for text in df['text'])
    total_processed_tokens_stem = sum(len(tokens) for tokens in df['processed_stem'])
    total_processed_tokens_lemma = sum(len(tokens) for tokens in df['processed_lemma'])

    print(f"Total characters in original text: {total_original_chars}")
    print(f"Total tokens after stemming: {total_processed_tokens_stem}")
    print(f"Total tokens after lemmatization: {total_processed_tokens_lemma}")

    # Processing efficiency
    avg_reduction_stem = (total_original_chars - total_processed_tokens_stem) / total_original_chars * 100
    avg_reduction_lemma = (total_original_chars - total_processed_tokens_lemma) / total_original_chars * 100

    print(f"Average reduction")


