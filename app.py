from flask import Flask, render_template, request
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import matplotlib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

matplotlib.use('Agg')

# Download required NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Combine NLTK stopwords with custom domain-specific stop words
STOP_WORDS = set(stopwords.words('english')).union({
    # Common email/support words
    'hi', 'hello', 'dear', 'regards', 'sincerely', 'please', 'thank', 'thanks',
     'calling', 'called', 'caller', 'calls',
     'notes', 'noted', 'noting',
    'issues', 'problem', 'problems',
    'customer', 'customers', 'client', 'clients',
    'number', 'numbers', 'num', 'destination',
    'wait', 'waiting', 'waited',
    'know', 'known', 'knowing',
    'still', 'yet', 'let','seems','sure','vcapsc','time','back','mcculler'

    # Technical terms
    'hangup', 'hangs', 'hung',
    'errors',
    'status', 'statuses',
    'log', 'logs', 'logged',
    'extcallid','wa','partnerid',

    # Time-related
    'am', 'pm', 'today', 'tomorrow', 'yesterday',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
    'september', 'october', 'november', 'december',

    # Common verbs
    'get', 'got', 'getting',
    'try', 'tried', 'trying',
    'see', 'saw', 'seen', 'seeing',
    'use', 'used', 'using',
    'need', 'needed', 'needs',
    'want', 'wanted', 'wants',
    'look', 'looked', 'looking','set','able','ive'

    # Common prepositions and articles
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'from', 'up', 'down', 'over', 'under', 'again','sla','ha','wa','would'
})


def process_text(text):
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Convert to lowercase and split by spaces
    words = text.lower().split()

    # Remove special characters, numbers, and common patterns
    cleaned_words = []
    for word in words:
        # Skip pure numbers or strings that are mostly numbers
        if re.match(r'^[\d\W]*$', word) or \
                sum(c.isdigit() for c in word) / len(word) > 0.3:
            continue

        # Remove special characters
        word = re.sub(r'[^a-z]', '', word)

        # Skip if too short or empty
        if len(word) <= 2:
            continue

        # Skip if it contains repeating characters (like 'aaaaaa')
        if re.search(r'(.)\1{2,}', word):
            continue

        cleaned_words.append(word)

    # Lemmatize words (combines similar words)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in cleaned_words]

    # Filter out stop words
    filtered_words = [word for word in lemmatized_words
                      if word and word not in STOP_WORDS]

    return filtered_words


def generate_word_cloud(word_freq):
    # Create and configure the WordCloud object
    wordcloud = WordCloud(
        width=3200,
        height=1600,
        background_color='white',
        max_words=100,
        prefer_horizontal=0.7
    )

    # Generate the word cloud
    wordcloud.generate_from_frequencies(word_freq)

    # Create a matplotlib figure
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Save the plot to a base64 string
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url


def generate_bar_chart(word_freq, top_n=20):
    # Get top N words
    top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n])

    # Create bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_words)), list(top_words.values()))
    plt.xticks(range(len(top_words)), list(top_words.keys()), rotation=45, ha='right')
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.tight_layout()

    # Save the plot to a base64 string
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']

        # Process the text
        words = process_text(text)

        # Count word frequencies
        word_freq = Counter(words)

        # Generate visualizations
        wordcloud_img = generate_word_cloud(word_freq)
        barchart_img = generate_bar_chart(word_freq)

        return render_template(
            'result.html',
            wordcloud=wordcloud_img,
            barchart=barchart_img,
            word_freq=dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20])
        )

    return render_template('index.html')


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    import os

    if not os.path.exists('templates'):
        os.makedirs('templates')

    # Create templates
    with open('templates/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Word Cloud Generator</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        textarea { width: 100%; height: 200px; margin: 20px 0; }
        button { padding: 10px 20px; }
    </style>
</head>
<body>
    <h1>Word Cloud Generator</h1>
    <form method="POST">
        <textarea name="text" placeholder="Enter your text here..."></textarea>
        <br>
        <button type="submit">Generate Visualizations</button>
    </form>
</body>
</html>
""")

    with open('templates/result.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Word Cloud Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .visualization { margin: 20px 0; }
        table { border-collapse: collapse; width: 100%; max-width: 600px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Results</h1>

    <div class="visualization">
        <h2>Word Cloud</h2>
        <img src="data:image/png;base64,{{ wordcloud }}" alt="Word Cloud">
    </div>

    <div class="visualization">
        <h2>Top 20 Words</h2>
        <img src="data:image/png;base64,{{ barchart }}" alt="Bar Chart">
    </div>

    <div class="visualization">
        <h2>Word Frequencies</h2>
        <table>
            <tr>
                <th>Word</th>
                <th>Frequency</th>
            </tr>
            {% for word, freq in word_freq.items() %}
            <tr>
                <td>{{ word }}</td>
                <td>{{ freq }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <br>
    <a href="/">Generate Another</a>
</body>
</html>
""")

    app.run(debug=True)