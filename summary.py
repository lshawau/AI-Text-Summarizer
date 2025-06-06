# from imports import *
import os
import sys
import time
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import requests
import json
import string
import uuid
import spacy
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import tiktoken
import matplotlib

# Preset configuration
PRESETS_FILE = "prompt_presets.json"
DEFAULT_PRESETS = {
    "built_in": {
        "default": {
            "name": "Default Summary",
            "prompt": "Please provide a concise summary of the following text, focusing on the main points",
            "category": "general"
        },
        "professional": {
            "name": "Professional Email",
            "prompt": "Analyze this text for professional tone and clarity, suggesting improvements",
            "category": "business"
        },
        "academic": {
            "name": "Academic Writing",
            "prompt": "Convert this text into a formal academic style, maintaining key information",
            "category": "education"
        }
    },
    "custom": {}
}

# Dark mode configuration
config = {
    "bg_color": "#1e1e2e",
    "fg_color": "#ffffff",
    "accent_color": "#ffd700",
    "entry_bg_color": "#2d2d3d",
    "button_bg_color": "#2d2d3d",
    "button_fg_color": "#ffffff"
}

class NLPPipeline:
    """
    NLP Pipeline that processes text through sequential stages,
    storing intermediate results for display in analysis tabs.
    """
    
    def __init__(self, raw_text):
        self.stages = {
            'raw': raw_text,
            'normalized': None,
            'tokens': None,
            'lemmatized': None,
            'pos_tagged': None,
            'entities': None,
            'key_terms': None,
            'sentiment': None,
            'pronunciation': None,
            'features': None
        }
        self.metadata = {}
        self.process_all_stages()
    
    def process_all_stages(self):
        """Process text through all NLP stages sequentially"""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('cmudict', quiet=True)
            
            # Stage 1: Text Normalization
            self.stages['normalized'] = self._normalize_text(self.stages['raw'])
            
            # Stage 2: Tokenization
            self.stages['tokens'] = self._tokenize_text(self.stages['normalized'])
            
            # Stage 3: Lemmatization & Stemming
            self.stages['lemmatized'] = self._lemmatize_text(self.stages['tokens'])
            
            # Stage 4: POS Tagging
            self.stages['pos_tagged'] = self._pos_tag_text(self.stages['lemmatized'])
            
            # Stage 5: Named Entity Recognition
            self.stages['entities'] = self._extract_entities(self.stages['pos_tagged'])
            
            # Stage 6: Key Terms Analysis
            self.stages['key_terms'] = self._analyze_key_terms(self.stages['lemmatized'])
            
            # Stage 7: Sentiment Analysis
            self.stages['sentiment'] = self._analyze_sentiment(self.stages['raw'])
            
            # Stage 8: Pronunciation Analysis
            self.stages['pronunciation'] = self._analyze_pronunciation(self.stages['tokens'])
            
            # Stage 9: Feature Engineering
            self.stages['features'] = self._extract_features(self.stages['raw'])
            
        except Exception as e:
            print(f"Pipeline processing error: {e}")
    
    def _normalize_text(self, text):
        """Stage 1: Text normalization"""
        steps = {
            'original': text,
            'lowercase': text.lower(),
            'no_punctuation': None,
            'no_stopwords': None
        }
        
        # Remove punctuation
        steps['no_punctuation'] = steps['lowercase'].translate(str.maketrans("", "", string.punctuation))
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            words = steps['no_punctuation'].split()
            steps['no_stopwords'] = ' '.join([word for word in words if word not in stop_words])
        except:
            steps['no_stopwords'] = steps['no_punctuation']
        
        return steps
    
    def _tokenize_text(self, normalized_data):
        """Stage 2: Tokenization"""
        text = normalized_data['original']  # Use original text for better tokenization
        
        # Basic word tokenization
        basic_tokens = text.split()
        
        # NLTK tokenization
        nltk_tokens = word_tokenize(text)
        
        # OpenAI tokenization (if available)
        openai_tokens = None
        token_details = []
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            openai_tokens = enc.encode(text)
            
            for token in openai_tokens:
                token_piece = enc.decode([token])
                display_piece = token_piece.replace('\n', '\\n').replace('\t', '\\t')
                token_details.append({
                    'text': display_piece,
                    'id': token,
                    'length': len(token_piece)
                })
        except:
            pass
        
        return {
            'basic': basic_tokens,
            'nltk': nltk_tokens,
            'openai': openai_tokens,
            'openai_details': token_details,
            'text': text
        }
    
    def _lemmatize_text(self, token_data):
        """Stage 3: Lemmatization and stemming"""
        tokens = token_data['nltk']
        
        try:
            lemmatizer = WordNetLemmatizer()
            stemmer = PorterStemmer()
            
            processed_words = []
            for word in tokens:
                if word.strip() and word not in string.punctuation:
                    lemma = lemmatizer.lemmatize(word)
                    stem = stemmer.stem(word)
                    processed_words.append({
                        'original': word,
                        'lemma': lemma,
                        'stem': stem
                    })
            
            return {
                'processed_words': processed_words,
                'lemmatized_text': ' '.join([w['lemma'] for w in processed_words]),
                'stemmed_text': ' '.join([w['stem'] for w in processed_words])
            }
        except Exception as e:
            return {'error': str(e), 'processed_words': []}
    
    def _pos_tag_text(self, lemma_data):
        """Stage 4: Part of Speech tagging"""
        try:
            # Use original tokens for POS tagging
            tokens = [w['original'] for w in lemma_data.get('processed_words', [])]
            if not tokens:
                tokens = word_tokenize(self.stages['raw'])
            
            pos_tags = pos_tag(tokens)
            
            return {
                'pos_tags': pos_tags,
                'tag_explanations': {
                    'NN': 'Noun (singular)',
                    'NNS': 'Noun (plural)',
                    'NNP': 'Proper noun (singular)',
                    'NNPS': 'Proper noun (plural)',
                    'VB': 'Verb (base form)',
                    'VBD': 'Verb (past tense)',
                    'VBG': 'Verb (gerund)',
                    'VBN': 'Verb (past participle)',
                    'VBP': 'Verb (present)',
                    'VBZ': 'Verb (3rd person singular)',
                    'JJ': 'Adjective',
                    'JJR': 'Adjective (comparative)',
                    'JJS': 'Adjective (superlative)',
                    'RB': 'Adverb',
                    'RBR': 'Adverb (comparative)',
                    'RBS': 'Adverb (superlative)',
                    'IN': 'Preposition/Subordinating conjunction',
                    'DT': 'Determiner',
                    'CC': 'Coordinating conjunction',
                    'PRP': 'Personal pronoun',
                    'MD': 'Modal verb'
                }
            }
        except Exception as e:
            return {'error': str(e), 'pos_tags': []}
    
    def _extract_entities(self, pos_data):
        """Stage 5: Named Entity Recognition"""
        try:
            pos_tags = pos_data.get('pos_tags', [])
            if not pos_tags:
                return {'error': 'No POS tags available', 'entities': []}
            
            named_entities = ne_chunk(pos_tags)
            
            entities = []
            for chunk in named_entities:
                if hasattr(chunk, 'label'):
                    entity = ' '.join(c[0] for c in chunk)
                    entities.append({
                        'text': entity,
                        'label': chunk.label()
                    })
            
            entity_types = {
                'PERSON': 'Names of people',
                'ORGANIZATION': 'Companies, institutions, government agencies',
                'GPE': 'Geo-Political Entities (cities, states, countries)',
                'LOCATION': 'Natural locations (mountains, rivers, etc)',
                'FACILITY': 'Buildings, airports, highways, bridges',
                'PRODUCT': 'Products, objects, vehicles',
                'EVENT': 'Historical events, hurricanes, wars',
                'WORK_OF_ART': 'Titles of books, songs',
                'LAW': 'Named documents made into laws',
                'LANGUAGE': 'Any named language',
                'DATE': 'Absolute or relative dates',
                'TIME': 'Times smaller than a day',
                'PERCENT': 'Percentage',
                'MONEY': 'Monetary values',
                'QUANTITY': 'Measurements'
            }
            
            return {
                'entities': entities,
                'entity_types': entity_types,
                'named_entities': named_entities
            }
        except Exception as e:
            return {'error': str(e), 'entities': []}
    
    def _analyze_key_terms(self, lemma_data):
        """Stage 6: Key terms analysis"""
        try:
            # Get lemmatized words
            processed_words = lemma_data.get('processed_words', [])
            if not processed_words:
                return {'error': 'No processed words available', 'terms': []}
            
            # Get lemmatized content words
            lemmas = [w['lemma'].lower() for w in processed_words if w['lemma'].isalpha()]
            
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            content_words = [word for word in lemmas if word not in stop_words]
            
            # Calculate frequencies
            word_freq = Counter(content_words)
            
            return {
                'word_frequencies': word_freq.most_common(15),
                'total_words': len(lemmas),
                'content_words': len(content_words),
                'unique_words': len(set(content_words))
            }
        except Exception as e:
            return {'error': str(e), 'terms': []}
    
    def _analyze_sentiment(self, text):
        """Stage 7: Sentiment analysis"""
        try:
            # VADER sentiment analysis
            sia = SentimentIntensityAnalyzer()
            
            # Overall sentiment
            overall_scores = sia.polarity_scores(text)
            
            # Sentence-level sentiment
            sentences = text.split('.')
            sentence_sentiments = []
            
            for sentence in sentences:
                if sentence.strip():
                    scores = sia.polarity_scores(sentence.strip())
                    sentiment_label = "positive" if scores['compound'] > 0 else "negative" if scores['compound'] < 0 else "neutral"
                    sentence_sentiments.append({
                        'text': sentence.strip(),
                        'scores': scores,
                        'label': sentiment_label
                    })
            
            return {
                'overall_scores': overall_scores,
                'sentence_sentiments': sentence_sentiments,
                'score_explanations': {
                    'pos': 'Positive sentiment score (0.0 to 1.0)',
                    'neu': 'Neutral sentiment score (0.0 to 1.0)',
                    'neg': 'Negative sentiment score (0.0 to 1.0)',
                    'compound': 'Compound score (-1.0 to 1.0, normalized weighted composite)'
                }
            }
        except Exception as e:
            return {'error': str(e), 'overall_scores': {}}
    
    def _analyze_pronunciation(self, token_data):
        """Stage 8: Pronunciation analysis"""
        try:
            from nltk.corpus import cmudict
            
            prondict = cmudict.dict()
            tokens = token_data.get('nltk', [])
            
            pronunciations = []
            
            for word in tokens:
                word_lower = word.lower()
                if word_lower.isalpha() and word_lower in prondict:
                    pron = prondict[word_lower][0]
                    
                    # Format pronunciation with stress markers
                    formatted_pron = []
                    for phoneme in pron:
                        if phoneme[-1].isdigit():
                            stress = phoneme[-1]
                            base = phoneme[:-1]
                            if stress == '1':
                                formatted_pron.append(f"ˈ{base}")  # Primary stress
                            elif stress == '2':
                                formatted_pron.append(f"ˌ{base}")  # Secondary stress
                            else:
                                formatted_pron.append(base)  # No stress
                        else:
                            formatted_pron.append(phoneme)
                    
                    pronunciations.append({
                        'word': word,
                        'pronunciation': ' '.join(formatted_pron),
                        'raw_phonemes': pron
                    })
            
            stress_guide = {
                '0': 'No stress',
                '1': 'Primary stress',
                '2': 'Secondary stress'
            }
            
            return {
                'pronunciations': pronunciations,
                'stress_guide': stress_guide
            }
        except Exception as e:
            return {'error': str(e), 'pronunciations': []}
    
    def _extract_features(self, text):
        """Stage 9: Feature engineering"""
        try:
            # Get required components
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            named_entities = ne_chunk(pos_tags)
            sia = SentimentIntensityAnalyzer()
            
            # Track entities for lookup
            entity_words = set()
            for chunk in named_entities:
                if hasattr(chunk, 'label'):
                    entity_words.update(word.lower() for word, tag in chunk.leaves())
            
            # Get OpenAI tokenizer
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
            except:
                enc = None
            
            # Extract features for each word
            features = []
            for word, (_, pos) in zip(tokens, pos_tags):
                # Calculate basic features
                length = len(word)
                capitals = sum(1 for c in word if c.isupper())
                punct = sum(1 for c in word if c in string.punctuation)
                sentiment = sia.polarity_scores(word)['compound']
                is_entity = "Yes" if word.lower() in entity_words else "No"
                
                # Get OpenAI token ID (if single token)
                token_id = "..."
                if enc:
                    try:
                        tokens_encoded = enc.encode(word)
                        token_id = tokens_encoded[0] if len(tokens_encoded) == 1 else "..."
                    except:
                        pass
                
                features.append({
                    'word': word,
                    'length': length,
                    'capitals': capitals,
                    'punctuation': punct,
                    'sentiment': sentiment,
                    'pos': pos,
                    'is_entity': is_entity,
                    'token_id': token_id
                })
            
            feature_explanations = {
                'length': 'Character count of the word',
                'capitals': 'Number of capital letters',
                'punctuation': 'Number of punctuation marks',
                'sentiment': 'VADER sentiment score (-1 to 1)',
                'pos': 'Part of speech tag',
                'is_entity': 'Whether word is part of a named entity',
                'token_id': 'OpenAI token ID (if single token)'
            }
            
            return {
                'features': features,
                'explanations': feature_explanations
            }
        except Exception as e:
            return {'error': str(e), 'features': []}


class TextSummaryApp:
    def format_features_row(self, word, length, capitals, punct, sentiment, pos, is_entity, token_id):
        """Helper function to format a single row in the features table"""
        # Convert all inputs to strings with proper formatting
        word_str = "{:<25}".format(str(word))
        length_str = "{:>4}".format(str(length))
        capitals_str = "{:>3}".format(str(capitals))
        punct_str = "{:>3}".format(str(punct))
        # Handle sentiment separately since it needs float formatting
        if isinstance(sentiment, (int, float)):
            sentiment_str = "{:>6.3f}".format(float(sentiment))
        else:
            sentiment_str = "{:>6}".format(str(sentiment))
        pos_str = "{:^6}".format(str(pos))
        entity_str = "{:^5}".format(str(is_entity))
        token_str = "{:>8}".format(str(token_id))
        
        # Combine all parts with proper spacing
        return (word_str + length_str + "  " + capitals_str + "  " + 
                punct_str + "  " + sentiment_str + "  " + pos_str + "  " + 
                entity_str + "  " + token_str)

    def __init__(self, root):
        self.root = root
        self.root.title("AI Text Summarizer - Advanced Analysis")
        self.root.geometry("1000x800")
        
        # Configure dark mode styles
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TLabel', background=config["bg_color"], foreground=config["fg_color"])
        style.configure('TEntry', fieldbackground=config["entry_bg_color"], foreground=config["fg_color"])
        style.configure('TButton', background=config["button_bg_color"], foreground=config["button_fg_color"])
        style.map('TButton',
                 foreground=[('active', config["accent_color"])],
                 background=[('active', config["button_bg_color"])])  # Keep same background on hover
        style.configure('TFrame', background=config["bg_color"])
        style.configure('TLabelframe', background=config["bg_color"])
        style.configure('TLabelframe.Label', background=config["bg_color"], foreground=config["fg_color"])
             
        # Configure gold progress bar style
        style.configure('Gold.Horizontal.TProgressbar',
                       troughcolor=config["bg_color"],
                       background=config["accent_color"])
        
        # Configure scrollbar style
        style.configure('TScrollbar', 
                       background=config["button_bg_color"],
                       troughcolor=config["bg_color"],
                       borderwidth=0,
                       arrowcolor=config["fg_color"])
        
        # Configure root background
        self.root.configure(bg=config["bg_color"])
        
        # Initialize presets
        self.presets = DEFAULT_PRESETS.copy()
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10", style='TFrame')
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # API Settings frame
        api_frame = ttk.LabelFrame(main_frame, text="API Settings", padding="5")
        api_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10)) 

        # API URL
        url_frame = ttk.Frame(api_frame)
        url_frame.pack(fill=tk.X, expand=True)
        ttk.Label(url_frame, text="API URL:").pack(side=tk.LEFT, padx=(0, 5))
        self.api_url = tk.StringVar(value="http://localhost:1234")
        self.api_entry = ttk.Entry(url_frame, textvariable=self.api_url, width=40, style='TEntry')
        self.api_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Model selection
        model_frame = ttk.Frame(api_frame)
        model_frame.pack(fill=tk.X, expand=True, pady=(5, 0))
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 5))
        self.available_models = []
        self.model_dropdown = ttk.Combobox(model_frame, values=self.available_models, state="readonly", width=30)
        self.model_dropdown.pack(side=tk.LEFT, padx=(0, 5)) 

        # Refresh models button
        self.refresh_button = ttk.Button(model_frame, text="Refresh Models", command=self.refresh_models)
        self.refresh_button.pack(side=tk.LEFT, padx=5)  

        # API Key
        key_frame = ttk.Frame(api_frame)
        key_frame.pack(fill=tk.X, expand=True, pady=(5, 0))
        ttk.Label(key_frame, text="API Key:").pack(side=tk.LEFT, padx=(0, 5))
        self.api_key = ttk.Entry(key_frame, width=40, style='TEntry', show="*")
        self.api_key.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Initialize available models
        self.detect_running_models()

        # Input label with count
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Label(input_frame, text="Input Text:").pack(side=tk.LEFT)
        self.input_count = ttk.Label(input_frame, text="(0 chars, 0 words)")
        self.input_count.pack(side=tk.RIGHT)
        
        # Input text area with adjustable height
        self.input_text = tk.Text(main_frame, height=10, width=100, wrap=tk.WORD,
                                 bg=config["entry_bg_color"], fg=config["fg_color"],
                                 insertbackground=config["fg_color"])
        self.input_text.bind('<<Modified>>', self.update_input_count)
        self.input_text.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbar to input
        input_scroll = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.input_text.yview,
                                    style='TScrollbar')
        input_scroll.grid(row=2, column=1, sticky=(tk.N, tk.S))
        self.input_text['yscrollcommand'] = input_scroll.set
        
        # Prompt frame with buttons
        prompt_frame = ttk.Frame(main_frame)
        prompt_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 5))
        
        # Left side - label and buttons
        prompt_label = ttk.Label(prompt_frame, text="System Prompt:", font=("TkDefaultFont", 9, "bold"))
        prompt_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Preset buttons
        self.preset_buttons_frame = ttk.Frame(prompt_frame)
        self.preset_buttons_frame.pack(side=tk.LEFT)
        
        # Load and create preset buttons
        self.load_presets()
        self.create_preset_buttons()
        
        # Manage presets button
        ttk.Button(prompt_frame, text="New Preset", command=self.create_new_preset).pack(side=tk.RIGHT, padx=2)
        ttk.Button(prompt_frame, text="Manage", command=self.manage_presets).pack(side=tk.RIGHT, padx=2)
        
        # Custom prompt input
        self.prompt_text = tk.Text(main_frame, height=3, width=100, wrap=tk.WORD,
                                  bg=config["entry_bg_color"], fg=config["fg_color"],
                                  insertbackground=config["fg_color"])
        self.prompt_text.grid(row=4, column=0, sticky=(tk.W, tk.E))
        
        # Add placeholder text for prompt with dimmed color
        self.prompt_placeholder = "Enter custom instructions for the AI (e.g., 'Summarize this text focusing on key events' or 'Create a bullet-point summary')"
        self.prompt_text.insert("1.0", self.prompt_placeholder)
        self.prompt_text.config(fg="#666666")  # Dimmed text color
        
        # Bind focus events for placeholder behavior
        self.prompt_text.bind("<FocusIn>", self.on_prompt_focus_in)
        self.prompt_text.bind("<FocusOut>", self.on_prompt_focus_out)
        
        # Add scrollbar to prompt
        prompt_scroll = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.prompt_text.yview,
                                     style='TScrollbar')
        prompt_scroll.grid(row=4, column=1, sticky=(tk.N, tk.S))
        self.prompt_text['yscrollcommand'] = prompt_scroll.set
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, pady=10)
        
        # Process button
        ttk.Button(button_frame, text="Generate Summary", command=self.process_text).pack(side=tk.LEFT, padx=5)
        
        # Analysis button (modified for pipeline)
        ttk.Button(button_frame, text="Show Advanced Analysis", command=self.show_analysis).pack(side=tk.LEFT, padx=5)
        
        # Create single output frame
        # Create header frame
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 5))
        
        # Create header label with word count
        self.header_label = ttk.Label(header_frame, text="Summary (0 words)")
        self.header_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Create resummarize button
        resummarize_btn = ttk.Button(
            header_frame, 
            text="Resummarize",
            command=self.resummarize
        )
        resummarize_btn.pack(side=tk.LEFT)
        
        # Create text widget with adjustable height
        self.output_text = tk.Text(main_frame, height=10, width=100, wrap=tk.WORD,
                                  bg=config["entry_bg_color"], fg=config["fg_color"],
                                  insertbackground=config["fg_color"])
        self.output_text.grid(row=7, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create scrollbar
        output_scroll = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.output_text.yview,
                                     style='TScrollbar')
        output_scroll.grid(row=7, column=1, sticky=(tk.N, tk.S))
        self.output_text['yscrollcommand'] = output_scroll.set
        
        # Configure grid weights for resizing
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)  # Make input text area expandable
        main_frame.rowconfigure(7, weight=1)  # Make output text area expandable
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

    def estimate_tokens(self, text):
        """
        Estimate token count using a simple word-piece approximation.
        This is a rough estimation - actual token count may vary by model.
        """
        # Split into words first
        words = text.split()
        token_count = 0
        
        for word in words:
            # Count punctuation as separate tokens
            punct_count = sum(1 for c in word if c in '.,!?;:()[]{}""\'')
            # Estimate word pieces (roughly 1.3 tokens per word on average)
            word_length = len(word)
            if word_length <= 2:
                token_count += 1
            else:
                # Longer words tend to be split into multiple tokens
                token_count += max(1, int(word_length / 3))
            token_count += punct_count
            
        return max(1, token_count)  # Minimum 1 token

    def count_words(self, text):
        # Split text into words and filter out empty strings
        words = [word for word in text.split() if word.strip()]
        return len(words)
        
    def update_input_count(self, event=None):
        if self.input_text.edit_modified():  # Check if content modified
            text = self.input_text.get("1.0", tk.END).strip()
            char_count = len(text)
            word_count = self.count_words(text)
            token_count = self.estimate_tokens(text)
            self.input_count.config(text=f"({char_count} chars, {word_count} words, ~{token_count} tokens)")
            self.input_text.edit_modified(False)  # Reset modified flag

    def update_word_count(self, text_widget):
        text = text_widget.get("1.0", tk.END).strip()
        word_count = self.count_words(text)
        self.header_label.config(text=f"Summary ({word_count} words)")

    def get_prompt(self, text):
        custom_prompt = self.prompt_text.get("1.0", tk.END).strip()
        default_prompt = f"Please provide a concise summary of the following text, focusing on the main points: {text}"
        
        if custom_prompt and custom_prompt != self.prompt_placeholder:
            return f"{custom_prompt}: {text}"
        return default_prompt

    def detect_running_models(self):
        """Detect running models using the OpenAI-like API endpoint"""
        try:
            response = requests.get(f"{self.api_url.get()}/v1/models", timeout=2)
            if response.status_code == 200:
                models_data = response.json()
                if isinstance(models_data, dict) and 'data' in models_data:
                    self.available_models = [model['id'] for model in models_data['data']]
                    self.model_dropdown['values'] = self.available_models
                    if self.available_models:
                        self.model_dropdown.set(self.available_models[0])
        except Exception as e:
            if not self.available_models:
                self.available_models = ["default"]
                self.model_dropdown['values'] = self.available_models
                self.model_dropdown.set("default")
            print(f"Error detecting models: {str(e)}")

    def refresh_models(self):
        """Refresh the list of available models"""
        self.detect_running_models()
        if self.available_models:
            messagebox.showinfo("Models Refreshed", f"Found {len(self.available_models)} models:\n" + "\n".join(self.available_models))
        else:
            messagebox.showwarning("Models Refreshed", "No models found. Check if the server is running.")

    def process_summary(self, prompt):
        try:
            # Clear previous output
            self.output_text.delete("1.0", tk.END)
            
            # Send request to AI instance
            response = requests.post(
                f"{self.api_url.get()}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "model": "local-model"
                }
            )
            
            if response.status_code == 200:
                # Extract the processed text from AI response
                processed_text = response.json()['choices'][0]['message']['content'].strip()
            else:
                processed_text = f"Error: Could not connect to AI service. Status code: {response.status_code}"
                
        except Exception as e:
            processed_text = f"Error processing text: {str(e)}"
            
        # Insert processed text
        self.output_text.insert("1.0", processed_text)
        
        # Update word count in header
        self.update_word_count(self.output_text)

    def resummarize(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            return
            
        prompt = self.get_prompt(text)
        self.process_summary(prompt)

    def on_prompt_focus_in(self, event):
        if self.prompt_text.get("1.0", "end-1c") == self.prompt_placeholder:
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.config(fg=config["fg_color"])

    def on_prompt_focus_out(self, event):
        if not self.prompt_text.get("1.0", "end-1c").strip():
            self.prompt_text.insert("1.0", self.prompt_placeholder)
            self.prompt_text.config(fg="#666666")

    def load_presets(self):
        """Load presets from file or create default"""
        try:
            if os.path.exists(PRESETS_FILE):
                with open(PRESETS_FILE, 'r') as f:
                    self.presets = json.load(f)
            else:
                self.presets = DEFAULT_PRESETS.copy()
                self.save_presets()
        except Exception as e:
            print(f"Error loading presets: {e}")
            self.presets = DEFAULT_PRESETS.copy()
            
    def save_presets(self):
        """Save presets to file"""
        try:
            with open(PRESETS_FILE, 'w') as f:
                json.dump(self.presets, f, indent=2)
        except Exception as e:
            print(f"Error saving presets: {e}")
            
    def create_preset_buttons(self):
        """Create buttons for all presets"""
        # Clear existing buttons
        for widget in self.preset_buttons_frame.winfo_children():
            widget.destroy()
            
        # Create buttons for built-in presets
        for preset_id, preset in self.presets["built_in"].items():
            ttk.Button(
                self.preset_buttons_frame,
                text=preset["name"],
                command=lambda p=preset["prompt"]: self.apply_preset(p)
            ).pack(side=tk.LEFT, padx=2)
            
        # Create buttons for custom presets
        for preset_id, preset in self.presets["custom"].items():
            ttk.Button(
                self.preset_buttons_frame,
                text=preset["name"],
                command=lambda p=preset["prompt"]: self.apply_preset(p)
            ).pack(side=tk.LEFT, padx=2)
            
    def apply_preset(self, prompt):
        """Apply selected preset to prompt text area"""
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", prompt)
        self.prompt_text.config(fg=config["fg_color"])
        
    def create_new_preset(self):
        """Open window to create new preset"""
        preset_window = tk.Toplevel(self.root)
        preset_window.title("Create New Preset")
        preset_window.geometry("500x300")
        preset_window.configure(bg=config["bg_color"])
        
        # Name input
        name_frame = ttk.Frame(preset_window)
        name_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(name_frame, text="Preset Name:").pack(side=tk.LEFT)
        name_entry = ttk.Entry(name_frame)
        name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Prompt input
        ttk.Label(preset_window, text="Prompt:").pack(anchor=tk.W, padx=10, pady=(10, 5))
        prompt_text = tk.Text(preset_window, height=8, wrap=tk.WORD,
                            bg=config["entry_bg_color"], fg=config["fg_color"])
        prompt_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Button frame
        button_frame = ttk.Frame(preset_window)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        def save_preset():
            name = name_entry.get().strip()
            prompt = prompt_text.get("1.0", tk.END).strip()
            
            if not name or not prompt:
                messagebox.showerror("Error", "Name and prompt are required")
                return
                
            # Create new preset
            preset_id = str(uuid.uuid4())
            self.presets["custom"][preset_id] = {
                "name": name,
                "prompt": prompt,
                "created": datetime.now().isoformat()
            }
            
            # Save and update UI
            self.save_presets()
            self.create_preset_buttons()
            preset_window.destroy()
            
        ttk.Button(button_frame, text="Save", command=save_preset).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=preset_window.destroy).pack(side=tk.RIGHT)
        
    def manage_presets(self):
        """Open window to manage presets"""
        manage_window = tk.Toplevel(self.root)
        manage_window.title("Manage Presets")
        manage_window.geometry("600x400")
        manage_window.configure(bg=config["bg_color"])
        
        # Create scrollable frame
        canvas = tk.Canvas(manage_window, bg=config["bg_color"])
        scrollbar = ttk.Scrollbar(manage_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Built-in presets section
        ttk.Label(scrollable_frame, text="Built-in Presets", font=("TkDefaultFont", 10, "bold")).pack(
            anchor=tk.W, padx=10, pady=(10, 5))
            
        for preset_id, preset in self.presets["built_in"].items():
            preset_frame = ttk.Frame(scrollable_frame)
            preset_frame.pack(fill=tk.X, padx=10, pady=2)
            
            ttk.Label(preset_frame, text=preset["name"]).pack(side=tk.LEFT)
            ttk.Button(
                preset_frame,
                text="Apply",
                command=lambda p=preset["prompt"]: self.apply_preset(p)
            ).pack(side=tk.RIGHT)
            
        # Custom presets section
        ttk.Label(scrollable_frame, text="Custom Presets", font=("TkDefaultFont", 10, "bold")).pack(
            anchor=tk.W, padx=10, pady=(20, 5))
            
        for preset_id, preset in self.presets["custom"].items():
            preset_frame = ttk.Frame(scrollable_frame)
            preset_frame.pack(fill=tk.X, padx=10, pady=2)
            
            ttk.Label(preset_frame, text=preset["name"]).pack(side=tk.LEFT)
            
            def delete_preset(pid=preset_id):
                if messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this preset?"):
                    del self.presets["custom"][pid]
                    self.save_presets()
                    self.create_preset_buttons()
                    manage_window.destroy()
                    self.manage_presets()
            
            ttk.Button(
                preset_frame,
                text="Delete",
                command=delete_preset
            ).pack(side=tk.RIGHT)
            
            ttk.Button(
                preset_frame,
                text="Apply",
                command=lambda p=preset["prompt"]: self.apply_preset(p)
            ).pack(side=tk.RIGHT, padx=5)
            
        # Pack canvas and scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def process_text(self):
        # Get input text
        text = self.input_text.get("1.0", tk.END).strip()
        
        if not text:
            return

        prompt = self.get_prompt(text)
        self.process_summary(prompt)

    def show_analysis(self):
        """Show advanced NLP pipeline analysis in a new window"""
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("No Text", "Please enter some text to analyze.")
            return
        
        # Create NLP pipeline
        try:
            pipeline = NLPPipeline(text)
        except Exception as e:
            messagebox.showerror("Pipeline Error", f"Error creating NLP pipeline: {str(e)}")
            return
            
        # Create analysis window
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("Advanced NLP Pipeline Analysis")
        analysis_window.geometry("1100x700")
        analysis_window.configure(bg=config["bg_color"])
        
        # Configure text widget style
        text_style = {
            "bg": config["entry_bg_color"],
            "fg": config["fg_color"],
            "insertbackground": config["fg_color"],
            "font": ("TkDefaultFont", 10)
        }
        
        # Configure notebook style
        style = ttk.Style(analysis_window)
        style.configure('TNotebook', background=config["bg_color"])
        style.configure('TNotebook.Tab', background=config["button_bg_color"],
                       foreground=config["fg_color"])
        style.map('TNotebook.Tab', background=[('selected', config["entry_bg_color"])])
        
        notebook = ttk.Notebook(analysis_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs in pipeline order
        self.create_normalization_tab(notebook, pipeline, text_style)
        self.create_tokenization_tab(notebook, pipeline, text_style)
        self.create_lemmatization_tab(notebook, pipeline, text_style)
        self.create_pos_tab(notebook, pipeline, text_style)
        self.create_ner_tab(notebook, pipeline, text_style)
        self.create_key_terms_tab(notebook, pipeline, text_style)
        self.create_sentiment_tab(notebook, pipeline, text_style)
        self.create_pronunciation_tab(notebook, pipeline, text_style)
        self.create_features_tab(notebook, pipeline, text_style)

    def create_normalization_tab(self, notebook, pipeline, text_style):
        """Stage 1: Text Normalization Tab"""
        norm_frame = ttk.Frame(notebook)
        notebook.add(norm_frame, text="1. Normalization")
        
        norm_text = tk.Text(norm_frame, height=10, width=80, **text_style)
        norm_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        norm_data = pipeline.stages['normalized']
        
        norm_text.insert(tk.END, "Text Normalization Analysis:\n")
        norm_text.insert(tk.END, "=" * 60 + "\n\n")
        norm_text.insert(tk.END, "Description:\n")
        norm_text.insert(tk.END, "How AI cleans and standardizes raw text for consistent processing.\n\n")
        norm_text.insert(tk.END, "Input from Previous Stage:\n")
        norm_text.insert(tk.END, "Raw user input text\n\n")
        norm_text.insert(tk.END, "Processing Steps:\n")
        norm_text.insert(tk.END, "=" * 60 + "\n\n")
        
        if norm_data and not norm_data.get('error'):
            norm_text.insert(tk.END, "1. Original Text:\n")
            preview = norm_data['original']
            norm_text.insert(tk.END, f"{preview}\n\n\n")
            
            norm_text.insert(tk.END, "2. Lowercase Conversion:\n")
            preview = norm_data['lowercase']
            norm_text.insert(tk.END, f"{preview}\n\n\n")
            
            norm_text.insert(tk.END, "3. Punctuation Removal:\n")
            preview = norm_data['no_punctuation']
            norm_text.insert(tk.END, f"{preview}\n\n\n")
            
            norm_text.insert(tk.END, "4. Stopword Removal:\n")
            preview = norm_data['no_stopwords']
            norm_text.insert(tk.END, f"{preview}\n\n\n")
            
            norm_text.insert(tk.END, "=" * 60 + "\n")
            norm_text.insert(tk.END, "Output to Next Stage:\n")
            norm_text.insert(tk.END, "Cleaned text variants ready for tokenization\n")
        else:
            norm_text.insert(tk.END, "Error in normalization process\n")
        
        norm_text.config(state='disabled')

    def create_tokenization_tab(self, notebook, pipeline, text_style):
        """Stage 2: Tokenization Tab"""
        token_frame = ttk.Frame(notebook)
        notebook.add(token_frame, text="2. Tokenization")
        
        token_text = tk.Text(token_frame, height=10, width=80, **text_style)
        token_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        token_data = pipeline.stages['tokens']
        norm_data = pipeline.stages['normalized']
        
        token_text.insert(tk.END, "Tokenization Analysis:\n")
        token_text.insert(tk.END, "=" * 60 + "\n\n")
        token_text.insert(tk.END, "Description:\n")
        token_text.insert(tk.END, "Breaking normalized text into individual units (tokens) that AI models can process.\n\n")
        token_text.insert(tk.END, "Input from Previous Stage:\n")
        if norm_data:
            preview = norm_data['original'][:150] + ('...' if len(norm_data['original']) > 150 else '')
            token_text.insert(tk.END, f"Normalized text: {preview}\n\n")
        token_text.insert(tk.END, "Processing Methods:\n")
        token_text.insert(tk.END, "=" * 60 + "\n\n")
        
        if token_data and not token_data.get('error'):
            # Basic tokenization
            token_text.insert(tk.END, "Basic Word Tokenization:\n")
            basic_tokens = token_data['basic']
            token_text.insert(tk.END, " | ".join(basic_tokens))
            if len(token_data['basic']) > 15:
                token_text.insert(tk.END, f" | ... (+{len(token_data['basic']) - 15} more)")
            token_text.insert(tk.END, f"\nTotal basic tokens: {len(token_data['basic'])}\n\n")
            
            # NLTK tokenization
            token_text.insert(tk.END, "NLTK Advanced Tokenization:\n")
            nltk_tokens = token_data['nltk']
            token_text.insert(tk.END, " | ".join(nltk_tokens))
            if len(token_data['nltk']) > 15:
                token_text.insert(tk.END, f" | ... (+{len(token_data['nltk']) - 15} more)")
            token_text.insert(tk.END, f"\nTotal NLTK tokens: {len(token_data['nltk'])}\n\n")
            
            # OpenAI tokenization
            if token_data['openai_details']:
                token_text.insert(tk.END, "OpenAI GPT Tokenization:\n")
                details = token_data['openai_details']
                for detail in details:
                    token_text.insert(tk.END, f"'{detail['text']}' → {detail['id']}\n")
                if len(token_data['openai_details']) > 10:
                    token_text.insert(tk.END, f"... (+{len(token_data['openai_details']) - 10} more tokens)\n")
                token_text.insert(tk.END, f"Total OpenAI tokens: {len(token_data['openai_details'])}\n\n")
            
            token_text.insert(tk.END, "=" * 60 + "\n")
            token_text.insert(tk.END, "Output to Next Stage:\n")
            token_text.insert(tk.END, "Token arrays ready for lemmatization and stemming\n")
        else:
            token_text.insert(tk.END, "Error in tokenization process\n")
        
        token_text.config(state='disabled')

    def create_lemmatization_tab(self, notebook, pipeline, text_style):
        """Stage 3: Lemmatization & Stemming Tab"""
        lemma_frame = ttk.Frame(notebook)
        notebook.add(lemma_frame, text="3. Word Forms")
        
        lemma_text = tk.Text(lemma_frame, height=10, width=80, **text_style)
        lemma_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        lemma_data = pipeline.stages['lemmatized']
        token_data = pipeline.stages['tokens']
        
        lemma_text.insert(tk.END, "Lemmatization & Stemming Analysis:\n")
        lemma_text.insert(tk.END, "=" * 60 + "\n\n")
        lemma_text.insert(tk.END, "Description:\n")
        lemma_text.insert(tk.END, "Reducing words to their base forms for consistent analysis and improved matching.\n\n")
        lemma_text.insert(tk.END, "Input from Previous Stage:\n")
        if token_data and token_data.get('nltk'):
            sample_tokens = token_data['nltk'][:8]
            lemma_text.insert(tk.END, f"NLTK tokens: {' | '.join(sample_tokens)}...\n\n")
        lemma_text.insert(tk.END, "Word Transformations:\n")
        lemma_text.insert(tk.END, "=" * 60 + "\n\n")
        
        if lemma_data and not lemma_data.get('error'):
            lemma_text.insert(tk.END, "Format: Original → Lemmatized → Stemmed\n\n")
            
            processed_words = lemma_data.get('processed_words', [])
            for word_data in processed_words:
                if (word_data['original'] != word_data['lemma'] or 
                    word_data['lemma'] != word_data['stem']):
                    lemma_text.insert(tk.END, 
                        f"{word_data['original']} → {word_data['lemma']} → {word_data['stem']}\n")
            
            if len(lemma_data.get('processed_words', [])) > 15:
                lemma_text.insert(tk.END, f"... (+{len(lemma_data['processed_words']) - 15} more transformations)\n")
            
            lemma_text.insert(tk.END, f"\nLemmatized Text Sample:\n")
            lemma_sample = lemma_data.get('lemmatized_text', '')
            lemma_text.insert(tk.END, f"{lemma_sample}\n\n")
            
            lemma_text.insert(tk.END, "=" * 60 + "\n")
            lemma_text.insert(tk.END, "Output to Next Stage:\n")
            lemma_text.insert(tk.END, "Normalized word forms ready for grammatical analysis\n")
        else:
            lemma_text.insert(tk.END, "Error in lemmatization process\n")
        
        lemma_text.config(state='disabled')

    def create_pos_tab(self, notebook, pipeline, text_style):
        """Stage 4: Part of Speech Tab"""
        pos_frame = ttk.Frame(notebook)
        notebook.add(pos_frame, text="4. Part of Speech")
        
        pos_text = tk.Text(pos_frame, height=10, width=80, **text_style)
        pos_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        pos_data = pipeline.stages['pos_tagged']
        lemma_data = pipeline.stages['lemmatized']
        
        pos_text.insert(tk.END, "Part of Speech Analysis:\n")
        pos_text.insert(tk.END, "=" * 60 + "\n\n")
        pos_text.insert(tk.END, "Description:\n")
        pos_text.insert(tk.END, "Identifying grammatical roles of words for understanding sentence structure.\n\n")
        pos_text.insert(tk.END, "Input from Previous Stage:\n")
        if lemma_data and lemma_data.get('processed_words'):
            sample_lemmas = [w['lemma'] for w in lemma_data['processed_words'][:8]]
            pos_text.insert(tk.END, f"Lemmatized words: {' | '.join(sample_lemmas)}...\n\n")
        
        if pos_data and not pos_data.get('error'):
            pos_text.insert(tk.END, "POS Tag Reference:\n")
            pos_text.insert(tk.END, "=" * 40 + "\n")
            explanations = pos_data.get('tag_explanations', {})
            for tag, explanation in list(explanations.items())[:10]:
                pos_text.insert(tk.END, f"{tag:>6}: {explanation}\n")
            pos_text.insert(tk.END, "\nTagged Words:\n")
            pos_text.insert(tk.END, "=" * 40 + "\n")
            
            pos_tags = pos_data.get('pos_tags', [])[:20]
            formatted_pos = []
            for word, tag in pos_tags:
                formatted_pos.append(f"{word}[{tag}]")
            pos_text.insert(tk.END, " ".join(formatted_pos))
            
            if len(pos_data.get('pos_tags', [])) > 20:
                pos_text.insert(tk.END, f"\n... (+{len(pos_data['pos_tags']) - 20} more tagged words)")
            
            pos_text.insert(tk.END, "\n\n" + "=" * 60 + "\n")
            pos_text.insert(tk.END, "Output to Next Stage:\n")
            pos_text.insert(tk.END, "Grammatically tagged words ready for entity recognition\n")
        else:
            pos_text.insert(tk.END, "Error in POS tagging process\n")
        
        pos_text.config(state='disabled')

    def create_ner_tab(self, notebook, pipeline, text_style):
        """Stage 5: Named Entity Recognition Tab"""
        ner_frame = ttk.Frame(notebook)
        notebook.add(ner_frame, text="5. Named Entities")
        
        ner_text = tk.Text(ner_frame, height=10, width=80, **text_style)
        ner_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ner_data = pipeline.stages['entities']
        pos_data = pipeline.stages['pos_tagged']
        
        ner_text.insert(tk.END, "Named Entity Recognition Analysis:\n")
        ner_text.insert(tk.END, "=" * 60 + "\n\n")
        ner_text.insert(tk.END, "Description:\n")
        ner_text.insert(tk.END, "Identifying important semantic entities (people, places, organizations, etc.)\n\n")
        ner_text.insert(tk.END, "Input from Previous Stage:\n")
        if pos_data and pos_data.get('pos_tags'):
            sample_tags = pos_data['pos_tags'][:6]
            sample_display = " ".join([f"{w}[{t}]" for w, t in sample_tags])
            ner_text.insert(tk.END, f"POS-tagged words: {sample_display}...\n\n")
        
        if ner_data and not ner_data.get('error'):
            # Entity type explanations
            ner_text.insert(tk.END, "Entity Types:\n")
            ner_text.insert(tk.END, "=" * 40 + "\n")
            entity_types = ner_data.get('entity_types', {})
            for entity_type, description in list(entity_types.items())[:8]:
                ner_text.insert(tk.END, f"{entity_type:>12}: {description}\n")
            
            # Found entities
            ner_text.insert(tk.END, "\nEntities Found:\n")
            ner_text.insert(tk.END, "=" * 40 + "\n")
            entities = ner_data.get('entities', [])
            if entities:
                for entity in entities:
                    ner_text.insert(tk.END, f"{entity['label']:>12}: {entity['text']}\n")
            else:
                ner_text.insert(tk.END, "No named entities detected in this text.\n")
            
            ner_text.insert(tk.END, "\n" + "=" * 60 + "\n")
            ner_text.insert(tk.END, "Output to Next Stage:\n")
            ner_text.insert(tk.END, "Entity-aware text ready for key terms frequency analysis\n")
        else:
            ner_text.insert(tk.END, "Error in named entity recognition process\n")
        
        ner_text.config(state='disabled')

    def create_key_terms_tab(self, notebook, pipeline, text_style):
        """Stage 6: Key Terms Tab"""
        terms_frame = ttk.Frame(notebook)
        notebook.add(terms_frame, text="6. Key Terms")
        
        terms_text = tk.Text(terms_frame, height=10, width=80, **text_style)
        terms_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        terms_data = pipeline.stages['key_terms']
        lemma_data = pipeline.stages['lemmatized']
        
        terms_text.insert(tk.END, "Key Terms Analysis:\n")
        terms_text.insert(tk.END, "=" * 60 + "\n\n")
        terms_text.insert(tk.END, "Description:\n")
        terms_text.insert(tk.END, "Frequency analysis of important content words after removing stopwords.\n\n")
        terms_text.insert(tk.END, "Input from Previous Stage:\n")
        if lemma_data and lemma_data.get('lemmatized_text'):
            preview = lemma_data['lemmatized_text'][:150] + ('...' if len(lemma_data['lemmatized_text']) > 150 else '')
            terms_text.insert(tk.END, f"Lemmatized text: {preview}\n\n")
        
        if terms_data and not terms_data.get('error'):
            terms_text.insert(tk.END, "Word Frequency Statistics:\n")
            terms_text.insert(tk.END, "=" * 40 + "\n")
            terms_text.insert(tk.END, f"Total words processed: {terms_data.get('total_words', 0)}\n")
            terms_text.insert(tk.END, f"Content words (no stopwords): {terms_data.get('content_words', 0)}\n")
            terms_text.insert(tk.END, f"Unique content words: {terms_data.get('unique_words', 0)}\n\n")
            
            terms_text.insert(tk.END, "Most Frequent Terms:\n")
            terms_text.insert(tk.END, "=" * 40 + "\n")
            word_frequencies = terms_data.get('word_frequencies', [])
            for word, count in word_frequencies:
                terms_text.insert(tk.END, f"{word:>15}: {count:>3} occurrences\n")
            
            terms_text.insert(tk.END, "\n" + "=" * 60 + "\n")
            terms_text.insert(tk.END, "Output to Next Stage:\n")
            terms_text.insert(tk.END, "Frequency-analyzed terms ready for sentiment analysis\n")
        else:
            terms_text.insert(tk.END, "Error in key terms analysis\n")
        
        terms_text.config(state='disabled')

    def create_sentiment_tab(self, notebook, pipeline, text_style):
        """Stage 7: Sentiment Tab"""
        sent_frame = ttk.Frame(notebook)
        notebook.add(sent_frame, text="7. Sentiment")
        
        sent_text = tk.Text(sent_frame, height=10, width=80, **text_style)
        sent_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        sent_data = pipeline.stages['sentiment']
        
        sent_text.insert(tk.END, "Sentiment Analysis:\n")
        sent_text.insert(tk.END, "=" * 60 + "\n\n")
        sent_text.insert(tk.END, "Description:\n")
        sent_text.insert(tk.END, "Analyzing emotional tone and opinion polarity using VADER sentiment analysis.\n\n")
        sent_text.insert(tk.END, "Input from Previous Stage:\n")
        sent_text.insert(tk.END, "Original text analyzed for emotional content\n\n")
        
        if sent_data and not sent_data.get('error'):
            # Overall sentiment
            overall_scores = sent_data.get('overall_scores', {})
            sent_text.insert(tk.END, "Overall Text Sentiment:\n")
            sent_text.insert(tk.END, "=" * 40 + "\n")
            sent_text.insert(tk.END, f"Positive: {overall_scores.get('pos', 0):.3f}\n")
            sent_text.insert(tk.END, f"Neutral:  {overall_scores.get('neu', 0):.3f}\n")
            sent_text.insert(tk.END, f"Negative: {overall_scores.get('neg', 0):.3f}\n")
            sent_text.insert(tk.END, f"Compound: {overall_scores.get('compound', 0):.3f}\n\n")
            
            # Sentence-level sentiment
            sentence_sentiments = sent_data.get('sentence_sentiments', [])[:5]
            if sentence_sentiments:
                sent_text.insert(tk.END, "Sentence-Level Analysis:\n")
                sent_text.insert(tk.END, "=" * 40 + "\n")
                for sent_info in sentence_sentiments:
                    sentence = sent_info['text'][:60] + ('...' if len(sent_info['text']) > 60 else '')
                    label = sent_info['label']
                    score = sent_info['scores']['compound']
                    sent_text.insert(tk.END, f"'{sentence}'\n")
                    sent_text.insert(tk.END, f"→ {label.upper()} (score: {score:.3f})\n\n")
            
            # Score explanations
            explanations = sent_data.get('score_explanations', {})
            sent_text.insert(tk.END, "Score Explanations:\n")
            sent_text.insert(tk.END, "=" * 40 + "\n")
            for score_type, explanation in explanations.items():
                sent_text.insert(tk.END, f"{score_type}: {explanation}\n")
            
            sent_text.insert(tk.END, "\n" + "=" * 60 + "\n")
            sent_text.insert(tk.END, "Output to Next Stage:\n")
            sent_text.insert(tk.END, "Sentiment-analyzed text ready for pronunciation analysis\n")
        else:
            sent_text.insert(tk.END, "Error in sentiment analysis\n")
        
        sent_text.config(state='disabled')

    def create_pronunciation_tab(self, notebook, pipeline, text_style):
        """Stage 8: Pronunciation Tab"""
        pron_frame = ttk.Frame(notebook)
        notebook.add(pron_frame, text="8. Pronunciation")
        
        pron_text = tk.Text(pron_frame, height=10, width=80, **text_style)
        pron_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        pron_data = pipeline.stages['pronunciation']
        token_data = pipeline.stages['tokens']
        
        pron_text.insert(tk.END, "Pronunciation & Diacritics Analysis:\n")
        pron_text.insert(tk.END, "=" * 60 + "\n\n")
        pron_text.insert(tk.END, "Description:\n")
        pron_text.insert(tk.END, "Phonetic transcription and stress pattern analysis using CMU Pronunciation Dictionary.\n\n")
        pron_text.insert(tk.END, "Input from Previous Stage:\n")
        if token_data and token_data.get('nltk'):
            sample_tokens = token_data['nltk'][:8]
            pron_text.insert(tk.END, f"NLTK tokens: {' | '.join(sample_tokens)}...\n\n")
        
        if pron_data and not pron_data.get('error'):
            # Stress guide
            stress_guide = pron_data.get('stress_guide', {})
            pron_text.insert(tk.END, "Stress Markers:\n")
            pron_text.insert(tk.END, "=" * 40 + "\n")
            for marker, meaning in stress_guide.items():
                pron_text.insert(tk.END, f"{marker}: {meaning}\n")
            pron_text.insert(tk.END, "\n")
            
            # Pronunciations
            pronunciations = pron_data.get('pronunciations', [])[:15]
            if pronunciations:
                pron_text.insert(tk.END, "Phonetic Transcriptions:\n")
                pron_text.insert(tk.END, "=" * 40 + "\n")
                for pron_info in pronunciations:
                    word = pron_info['word']
                    pronunciation = pron_info['pronunciation']
                    pron_text.insert(tk.END, f"{word:>12}: {pronunciation}\n")
                
                if len(pron_data.get('pronunciations', [])) > 15:
                    pron_text.insert(tk.END, f"... (+{len(pron_data['pronunciations']) - 15} more pronunciations)\n")
            else:
                pron_text.insert(tk.END, "No pronunciations found in CMU dictionary for these words.\n")
            
            pron_text.insert(tk.END, "\n" + "=" * 60 + "\n")
            pron_text.insert(tk.END, "Output to Next Stage:\n")
            pron_text.insert(tk.END, "Phonetically analyzed text ready for feature engineering\n")
        else:
            pron_text.insert(tk.END, "Error in pronunciation analysis\n")
        
        pron_text.config(state='disabled')

    def create_features_tab(self, notebook, pipeline, text_style):
        """Stage 9: Feature Engineering Tab"""
        feat_frame = ttk.Frame(notebook)
        notebook.add(feat_frame, text="9. Features")
        
        feat_text = tk.Text(feat_frame, height=10, width=80, **{
            **text_style,
            "font": ("Consolas", 9)  # Monospace font for table alignment
        })
        feat_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        feat_data = pipeline.stages['features']
        
        feat_text.insert(tk.END, "Feature Engineering Analysis:\n")
        feat_text.insert(tk.END, "=" * 60 + "\n\n")
        feat_text.insert(tk.END, "Description:\n")
        feat_text.insert(tk.END, "Converting text into numerical features that machine learning models can process.\n\n")
        feat_text.insert(tk.END, "Input from Previous Stage:\n")
        feat_text.insert(tk.END, "All previous analysis results combined for feature extraction\n\n")
        
        if feat_data and not feat_data.get('error'):
            # Feature explanations
            explanations = feat_data.get('explanations', {})
            feat_text.insert(tk.END, "Feature Explanations:\n")
            feat_text.insert(tk.END, "=" * 40 + "\n")
            for feature, explanation in explanations.items():
                feat_text.insert(tk.END, f"{feature}: {explanation}\n")
            feat_text.insert(tk.END, "\n" + "=" * 70 + "\n\n")
            
            # Create header
            header = self.format_features_row(
                "Text", "Len", "Cap", "Pct", "Sent", "POS", "Ent", "Token"
            )
            feat_text.insert(tk.END, header + "\n")
            feat_text.insert(tk.END, "=" * 70 + "\n")
            
            # Show features for first 15 words
            features = feat_data.get('features', [])[:15]
            for feat in features:
                row = self.format_features_row(
                    feat['word'], feat['length'], feat['capitals'], 
                    feat['punctuation'], feat['sentiment'], feat['pos'], 
                    feat['is_entity'], feat['token_id']
                )
                feat_text.insert(tk.END, row + "\n")
            
            if len(feat_data.get('features', [])) > 15:
                feat_text.insert(tk.END, f"... (+{len(feat_data['features']) - 15} more feature rows)\n")
            
            feat_text.insert(tk.END, "\n" + "=" * 60 + "\n")
            feat_text.insert(tk.END, "Output to Next Stage:\n")
            feat_text.insert(tk.END, "Numerical features ready for machine learning models\n")
        else:
            feat_text.insert(tk.END, "Error in feature extraction\n")
        
        feat_text.config(state='disabled')


def main():
    root = tk.Tk()
    app = TextSummaryApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
