# app.py
import os

from flask import Flask, render_template, request, jsonify
from collections import Counter, defaultdict
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import matplotlib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import json
from datetime import datetime
import redis
import json
from datetime import datetime
from flask_cors import CORS
import logging
from redis import Redis
from redis.exceptions import RedisError
import time

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

matplotlib.use('Agg')

# Download required NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

def get_redis_client():
    retries = 5
    while retries:
        try:
            client = Redis(host='redis', port=6379, db=0, decode_responses=True)
            client.ping()  # Test connection
            return client
        except RedisError as e:
            retries -= 1
            if not retries:
                raise
            time.sleep(1)  # Wait before retry

redis_client = get_redis_client()

# In-memory storage (replace with database in production)
word_contributions = defaultdict(list)  # {word: [(contributor, timestamp), ...]}
participants = {}  # {name: last_active_timestamp}

# Color palette
COLORS = {
    'primary': '#131C41',
    'secondary': '#A5B2E2',
    'background': '#F0F1F7',
    'text': '#333335',
    'accent1': '#906AE2',
    'accent2': '#28B8DB',
    'chart_colors': ['#845ADF', '#26BF94', '#23B7E5', '#F5B849', '#FA8231', '#C0392B']
}

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


def process_text(text, contributor):
    lemmatizer = WordNetLemmatizer()
    words = text.lower().split()

    cleaned_words = []
    for word in words:
        if re.match(r'^[\d\W]*$', word) or \
                sum(c.isdigit() for c in word) / len(word) > 0.3:
            continue

        word = re.sub(r'[^a-z]', '', word)

        if len(word) <= 2:
            continue

        if re.search(r'(.)\1{2,}', word):
            continue

        cleaned_words.append(word)

    lemmatized_words = [lemmatizer.lemmatize(word) for word in cleaned_words]
    filtered_words = [word for word in lemmatized_words
                      if word and word not in STOP_WORDS]

    # Record contributions
    timestamp = datetime.now().isoformat()
    for word in filtered_words:
        # Store word contributions as a list in Redis
        contribution = json.dumps({'contributor': contributor, 'timestamp': timestamp})
        redis_client.lpush(f'word:{word}', contribution)

    # Update participant's last active time
    redis_client.hset('participants', contributor, timestamp)

    return filtered_words


def generate_word_cloud(word_freq):
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color=COLORS['background'],
        colormap='viridis',
        max_words=100,
        prefer_horizontal=0.7
    )

    wordcloud.generate_from_frequencies(word_freq)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight',
                facecolor=COLORS['background'])
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()


def generate_bar_chart(word_freq, top_n=20):
    top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n])

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(top_words)), list(top_words.values()))

    # Color bars using the chart color palette
    for i, bar in enumerate(bars):
        bar.set_color(COLORS['chart_colors'][i % len(COLORS['chart_colors'])])

    plt.xticks(range(len(top_words)), list(top_words.keys()),
               rotation=45, ha='right')
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')

    # Style the chart
    plt.gca().set_facecolor(COLORS['background'])
    plt.gcf().set_facecolor(COLORS['background'])

    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight',
                facecolor=COLORS['background'])
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()


@app.route('/')
def index():
    return render_template('index.html', colors=COLORS)


@app.route('/contribute', methods=['POST'])
def contribute():
    try:
        data = request.json
        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No data received'}), 400

        name = data.get('name', '').strip()
        text = data.get('text', '').strip()

        if not name or not text:
            logger.error(f"Missing required fields: name={bool(name)}, text={bool(text)}")
            return jsonify({'error': 'Name and text are required'}), 400

        words = process_text(text, name)

        try:
            # Get overall word frequencies
            all_words = []
            for key in redis_client.keys('word:*'):
                word = key.decode('utf-8').split(':')[1]
                count = redis_client.llen(key)
                all_words.extend([word] * count)

            total_freq = Counter(all_words)

            # Check if we have any words before generating visualizations
            if not total_freq:
                return jsonify({
                    'wordcloud': None,
                    'barchart': None,
                    'frequencies': {},
                    'participants': [],
                    'message': 'No valid words found after filtering'
                })

            # Generate visualizations
            wordcloud = generate_word_cloud(total_freq)
            barchart = generate_bar_chart(total_freq)

            # Get participant list
            participants_data = {
                name.decode('utf-8'): timestamp.decode('utf-8')
                for name, timestamp in redis_client.hgetall('participants').items()
            }

            active_participants = sorted(
                [(name, timestamp) for name, timestamp in participants_data.items()],
                key=lambda x: x[1],
                reverse=True
            )

            return jsonify({
                'wordcloud': wordcloud,
                'barchart': barchart,
                'frequencies': dict(total_freq.most_common(20)),
                'participants': [p[0] for p in active_participants]
            })

        except redis.RedisError as e:
            logger.error(f"Redis error: {str(e)}")
            return jsonify({'error': 'Database error occurred'}), 500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': str(e)}), 500
@app.route('/health')
def health():
    try:
        # Check Redis connection
        redis_client.ping()
        return jsonify({
            'status': 'healthy',
            'instance': os.getenv('INSTANCE_NAME', 'unknown')
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'instance': os.getenv('INSTANCE_NAME', 'unknown')
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)