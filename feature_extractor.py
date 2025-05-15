import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from collections import Counter
import pickle

def feature(func):
    func._is_feature = True
    return func

class URLFeatureExtractor:
    def __init__(self):
        self.char_probs = None
        self.shortening_services = [
            'bit.ly', 'goo.gl', 'shorte.st', 'go2l.ink', 'x.co', 'ow.ly',
            'tinyurl.com', 'tr.im', 'is.gd', 'cli.gs', 'yfrog.com', 'migre.me',
            'ff.im', 'tiny.cc', 'url4.eu', 'twit.ac', 'su.pr', 'twurl.nl',
            'snipurl.com', 'short.to', 'budurl.com', 'ping.fm', 'post.ly',
            'just.as', 'bkite.com', 'snipr.com', 'fic.kr', 'loopt.us', 'doiop.com',
            'short.ie', 'kl.am', 'wp.me', 'rubyurl.com', 'om.ly', 'to.ly', 'bit.do'
        ]

    def fit(self, df: pd.DataFrame):
        self.configure_1gram_probs(df)

        # 2-gram bigram probability
        self.configure_bigram_probs(df)
        print(f"[DEBUG] Number of unique bigrams: {len(self.bigram_probs)}")

    def configure_1gram_probs(self, df):
        all_text = ''.join(df['url'].astype(str))
        # Character probability (1-gram)
        char_counter = Counter(all_text)
        total_chars = sum(char_counter.values())
        self.char_probs = {char: count / total_chars for char, count in char_counter.items()}

    def configure_bigram_probs(self, df):
        bigram_counter = Counter()
        for url in df['url'].astype(str):
            url = url + '$'  # Add end marker per URL
            bigrams = [url[i:i + 2] for i in range(len(url) - 1)]
            bigram_counter.update(bigrams)
        min_count = 3
        bigram_counter = Counter({bg: c for bg, c in bigram_counter.items() if c >= min_count})

        total_bigrams = sum(bigram_counter.values())
        self.bigram_probs = {bigram: count / total_bigrams for bigram, count in bigram_counter.items()}
        self.bigram_min_prob = min(self.bigram_probs.values()) if self.bigram_probs else 1e-6
        self.bigram_fallback_prob = self.bigram_min_prob / 10

    @feature
    def entropy(self, url: str) -> float:
        if not url or self.char_probs is None:
            return 0.0
        url_counter = Counter(url)
        total = sum(url_counter.values())
        entropy_value = 0.0
        for char, count in url_counter.items():
            p = self.char_probs.get(char, 1e-6)  # small prob for unseen chars
            entropy_value -= (count / total) * np.log2(p)
        return entropy_value

    @feature
    def is_ip(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if hostname is None:
                return False
            return bool(re.fullmatch(r'\d{1,3}(\.\d{1,3}){3}', hostname))
        except:
            return False

    @feature
    def url_length(self, url: str) -> int:
        return len(url)

    @feature
    def using_shortener(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            return any(service in domain for service in self.shortening_services)
        except:
            return False

    @feature
    def abnormal_double_slash(self, url: str) -> bool:
        pos = url.find('//')
        return pos > 7

    @feature
    def has_dash(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            return '-' in domain
        except:
            return False

    @feature
    def count_subdomains(self, url: str) -> int:
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if hostname is None:
                return 0
            parts = hostname.split('.')
            if len(parts) <= 2:
                return 0
            if parts[0] == 'www':
                return len(parts) - 3
            else:
                return len(parts) - 2
        except:
            return 0

    @feature
    def port_non_standard(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            if parsed.port is None:
                return False
            if parsed.scheme == 'http' and parsed.port != 80:
                return True
            if parsed.scheme == 'https' and parsed.port != 443:
                return True
            return False
        except:
            return False

    @feature
    def has_at_symbol(self, url: str) -> bool:
        return '@' in url

    @feature
    def is_https(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            return parsed.scheme.lower() == 'https'
        except:
            return False
    @feature
    def count_digits_in_path(self, url: str) -> int:
        try:
            path = urlparse(url).path
            return sum(char.isdigit() for char in path)
        except:
            return 0

    @feature
    def avg_word_length_in_path(self, url: str) -> float:
        try:
            path = urlparse(url).path
            words = [word for word in path.split('/') if word]
            if not words:
                return 0.0
            return sum(len(word) for word in words) / len(words)
        except:
            return 0.0

    @feature
    def longest_word_in_path(self, url: str) -> int:
        try:
            path = urlparse(url).path
            words = [word for word in path.split('/') if word]
            if not words:
                return 0
            return max(len(word) for word in words)
        except:
            return 0

    @feature
    def count_question_marks(self, url: str) -> int:
        return url.count('?')

    @feature
    def count_slashes_in_path(self, url: str) -> int:
        try:
            path = urlparse(url).path
            return path.count('/')
        except:
            return 0

    @feature
    def contains_dollar(self, url: str) -> int:
        return int('$' in url)

    @feature
    def contains_comma(self, url: str) -> int:
        return int(',' in url)

    @feature
    def contains_pipe(self, url: str) -> int:
        return int('|' in url)

    @feature
    def contains_semicolon(self, url: str) -> int:
        return int(';' in url)

    @feature
    def contains_whitespace(self, url: str) -> int:
        return int(' ' in url)

    @feature
    def hyphen_ratio(self, url: str) -> float:
        if not url:
            return 0.0
        return url.count('-') / len(url)

    @feature
    def bigram_entropy(self, url: str) -> float:
        if not url or self.bigram_probs is None:
            return 0.0

        # Append fake end-of-sequence marker
        url = url + '$'
        bigrams = [url[i:i + 2] for i in range(len(url) - 1)]

        if not bigrams:
            return 0.0

        bigram_counter = Counter(bigrams)
        total = sum(bigram_counter.values())
        entropy_value = 0.0

        for bigram, count in bigram_counter.items():
            p = self.bigram_probs.get(bigram, 1e-6)  # Small prob for unseen bigrams
            entropy_value -= (count / total) * np.log2(p)

        return entropy_value

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame()

        # Find all feature methods dynamically
        feature_methods = [method_name for method_name in dir(self)
                           if callable(getattr(self, method_name))
                           and hasattr(getattr(self, method_name), '_is_feature')]

        for method_name in feature_methods:
            method = getattr(self, method_name)
            feature_name = method_name  # Or you can customize name if you want
            features[feature_name] = df['url'].apply(method)

        return features

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)
