# AI-Text-Summarizer

A Python-based text analysis and summarization tool with a graphical interface. This tool provides comprehensive text analysis features for educational purposes including:
![Image](https://github.com/user-attachments/assets/63524bb3-d3f4-453f-9189-d9d9e4607f30)
- **Text summarization** with customizable prompts
  - **Examples of:**
    - **Advanced NLP Pipeline Analysis** with sequential processing stages
    - **Named Entity Recognition** with detailed entity classification
    - **Part of Speech tagging** with comprehensive grammatical analysis
    - **Sentiment analysis** using VADER sentiment analysis
    - **Token analysis** including OpenAI GPT tokenization and cost estimates
    - **Word form analysis** with lemmatization and stemming
    - **Pronunciation analysis** with phonetic transcriptions and stress patterns
    - **Text normalization** showing AI preprocessing steps
    - **Feature engineering** for machine learning applications

## Requirements

- Python 3.11 or earlier (recommended for best compatibility)
- Anaconda or Miniconda
- Internet connection for initial model downloads

## Installation (Conda Method - Recommended)

**⚠️ Important: Use conda to avoid pip dependency conflicts**

1. Clone this repository
2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate summaryai
```

3. **CRITICAL STEP** - Set up required models:
```bash
python setup_models.py
```

4. Run the application:
```bash
python summary.py          # Standard version
python summary_advanced.py # Advanced pipeline version
```

## Why Use the setup_models.py Script?

If you skip the model setup step, you'll encounter cryptic errors like:
- `LookupError: Resource punkt not found`
- `OSError: [E050] Can't find model 'en_core_web_sm'`
- And several other equally mysterious messages

The setup script automatically downloads and configures:
- **Spacy model**: en_core_web_sm
- **NLTK datasets**: punkt, averaged_perceptron_tagger, maxent_ne_chunker, words, vader_lexicon, wordnet, omw-1.4, cmudict

## Application Versions

### Standard Version (`summary.py`)
- Sequential NLP pipeline processing
- Shows how text transforms through each analysis stage
- Educational pipeline flow from raw text to ML features
- Professional NLP workflow demonstration

## Usage

### Basic Summarization
1. Enter or paste text in the input area
2. Optionally customize the AI prompt
3. Click "Generate Summary"
4. View results in the output area

### Advanced Text Analysis
1. Enter text for analysis
2. Click Show Advanced Analysis (pipeline)
3. Explore the tabbed analysis results:
   - **Normalization**: Text cleaning and standardization
   - **Tokenization**: Breaking text into processable units
   - **Word Forms**: Lemmatization and stemming transformations
   - **Part of Speech**: Grammatical role identification
   - **Named Entities**: People, places, organizations extraction
   - **Key Terms**: Frequency analysis of important words
   - **Sentiment**: Emotional tone and polarity analysis
   - **Pronunciation**: Phonetic transcriptions with stress markers
   - **Features**: Numerical feature extraction for machine learning

## Features

### Interface
- **Dark Mode Theme**: Professional dark interface design
- **Preset Management**: Built-in and custom prompt presets
- **Real-time Feedback**: Live character, word, and token counts
- **Model Selection**: Support for multiple AI models

### Analysis Capabilities
- **OpenAI Tokenization**: GPT-4 compatible token analysis with cost estimates
- **VADER Sentiment**: Comprehensive sentiment scoring (positive, negative, neutral, compound)
- **CMU Pronunciation**: Phonetic transcriptions using Carnegie Mellon University Pronunciation Dictionary
- **Named Entity Recognition**: 15+ entity types (PERSON, ORGANIZATION, GPE, etc.)
- **Advanced POS Tagging**: 20+ grammatical categories
- **Pipeline Visualization**: See how professional NLP systems process text step-by-step

### API Integration
- Compatible with OpenAI-like API endpoints
- Default local endpoint: `http://localhost:1234/v1/chat/completions`
- Configurable API URLs and authentication
- Model auto-detection and selection


## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure you've activated the conda environment (`conda activate summaryai`)
2. **Model Errors**: Run `python setup_models.py` to download required models
3. **API Errors**: Check that your local AI server is running on the configured port
4. **Performance Issues**: Large texts may take longer to process through the full pipeline

### Python Version Compatibility
- **Recommended**: Python 3.11 for best compatibility
- **Issues with Python > 3.11**: Some spaCy components may have compatibility issues

## File Structure
- `summary.py` - Application  
- `setup_models.py` - Model download and setup script
- `environment.yml` - Conda environment specification
- `prompt_presets.json` - User-customizable prompt templates

## Contributing

This tool is designed for educational and research purposes. Feel free to extend the analysis capabilities or improve the pipeline architecture.
