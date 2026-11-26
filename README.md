# NLP Text Preprocessing with NLTK

A hands-on walkthrough of core text preprocessing techniques for NLP, applied to a small sample of social media posts (Twitter, Facebook, LinkedIn) full of emojis, hashtags, slang, and punctuation noise.

## What's Covered

- *Dataset creation*: 10 sample posts saved to CSV with pandas
- *Lowercasing*
- *Punctuation, emoji, and special character removal*
- *Tokenization*: basic and advanced (contraction handling)
- *Stopword removal*: standard NLTK list plus custom social media stopwords (lol, omg, tbh, ...)
- *Stemming*: Porter vs Snowball
- *Lemmatization*: simple vs POS-aware (WordNet)
- *Stemming vs lemmatization comparison*
- *Reusable TextPreprocessor class*: full pipeline in one call
- *Analysis and visualization*: vocabulary reduction stats and word frequency plots

## Tech Stack

Python, NLTK, pandas, Matplotlib

## Setup

bash
git clone https://github.com/<your-username>/nlp-text-preprocessing-nltk.git
cd nlp-text-preprocessing-nltk
pip install nltk pandas matplotlib


Download the NLTK data (the script also does this automatically):

python
import nltk
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet",
            "omw-1.4", "averaged_perceptron_tagger"]:
    nltk.download(pkg)


## Usage

python
preprocessor = TextPreprocessor()

text = "OMG!!! Just had the BEST pizza ever 🍕🍕🍕 #foodie #yummy"
preprocessor.preprocess(text, method="lemmatize")
# ['get', 'best', 'pizza', 'ever', 'foodie', 'yummy']  (example output)


Methods: "stem", "lemmatize", or "none" (tokens only).

## Output Files

- social_media_posts.csv: raw sample data
- processed_social_media_posts.csv: stemmed, lemmatized, and token-only versions

## Key Takeaways

- Stemming is fast but can produce non-words (studies -> studi)
- Lemmatization is slower but returns real dictionary words, especially with POS tags
- Social media text needs custom stopwords and emoji handling beyond the defaults

## Next Steps

- Add TF-IDF / Bag-of-Words vectorization
- Sentiment analysis on the cleaned text
- Try spaCy for comparison

## Author

Your Name | [LinkedIn](https://www.linkedin.com/public-profile/settings/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_self_edit_contact_info%3BpPfzab%2BgR%2F6ZqRxxJ5%2BU7w%3D%3D)
