# Build a Semantic Book Recommender with LLMs – Full Course

This repo contains all of the code to complete the freeCodeCamp course, "Build a Semantic Book Recommender with LLMs – Full Course". This is a comprehensive book recommendation system that uses machine learning and natural language processing to provide intelligent book suggestions.

## Project Components

### 1. Data Exploration and Cleaning

- **File**: [`data-exploration.ipynb`](data-exploration.ipynb)
- Data analysis and visualization of the book dataset
- Handling missing values and data quality issues
- Feature engineering (age of book, word count in descriptions)
- Data filtering and preparation for downstream tasks
- Exports cleaned data to [`books_cleaned.csv`](books_cleaned.csv)

### 2. Text Classification

- **File**: [`text-classification.ipynb`](text-classification.ipynb)
- Zero-shot classification using LLMs to categorize books
- Classifies books as "Fiction" or "Non-fiction"
- Creates simple category mappings for better user filtering
- Exports categorized data to [`books_with_categories.csv`](books_with_categories.csv)

### 3. Sentiment Analysis and Emotion Detection

- **File**: [`sentiment-analysis.ipynb`](sentiment-analysis.ipynb)
- Extracts emotional content from book descriptions
- Analyzes emotions: anger, disgust, fear, joy, sadness, surprise, neutral
- Allows users to sort books by emotional tone and mood
- Exports emotion-enriched data to [`books_with_emotions.csv`](books_with_emotions.csv)

### 4. Vector Search and Semantic Similarity

- **File**: [`vector-search.ipynb`](vector-search.ipynb)
- Builds a vector database using semantic embeddings
- Enables natural language queries (e.g., "a book about revenge")
- Implements similarity search for book recommendations
- Creates tagged descriptions for enhanced searchability

### 5. Web Application Interface

- **File**: [`gradio-dashboard.py`](gradio-dashboard.py)
- Interactive web application built with Gradio
- User-friendly interface for book recommendations
- Combines all ML components into a single application
- **Jupyter Interface**: [`gradio.ipynb`](gradio.ipynb) for development and testing

## Data Files

- **Original Data**: Raw book dataset (downloaded from Kaggle)
- [`books_cleaned.csv`](books_cleaned.csv): Cleaned and preprocessed book data
- [`books_with_categories.csv`](books_with_categories.csv): Books with fiction/non-fiction classifications
- [`books_with_emotions.csv`](books_with_emotions.csv): Books with emotional analysis scores
- [`tagged_description.txt`](tagged_description.txt): Text file with tagged book descriptions
- [`cover-not-found.jpg`](cover-not-found.jpg): Placeholder image for missing book covers

## Configuration and Testing

- [`.env`](.env): Environment variables (contains OpenAI API key)
- [`requirements.txt`](requirements.txt): Complete list of Python dependencies
- `.gradio/`: Gradio configuration and certificates

## Technical Requirements

This project was created using Python 3.11. Required dependencies include:

### Core Data Science Libraries

- [pandas](https://pypi.org/project/pandas/) - Data manipulation and analysis
- [matplotlib](https://pypi.org/project/matplotlib/) - Data visualization
- [seaborn](https://pypi.org/project/seaborn/) - Statistical data visualization

### Machine Learning and NLP

- [transformers](https://pypi.org/project/transformers/) - Hugging Face transformers for NLP tasks
- [langchain-community](https://pypi.org/project/langchain-community/) - LangChain community integrations
- [langchain-opencv](https://pypi.org/project/langchain-opencv/) - Computer vision integration
- [langchain-chroma](https://pypi.org/project/langchain-chroma/) - Vector database integration

### Web Application and Development

- [gradio](https://pypi.org/project/gradio/) - Web interface for ML applications
- [notebook](https://pypi.org/project/notebook/) - Jupyter notebook environment
- [ipywidgets](https://pypi.org/project/ipywidgets/) - Interactive widgets for notebooks

### Utilities

- [kagglehub](https://pypi.org/project/kagglehub/) - Kaggle dataset integration
- [python-dotenv](https://pypi.org/project/python-dotenv/) - Environment variable management

## Setup Instructions

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   Create a `.env` file in the root directory with your OpenAI API key:

   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Data Setup**:
   Download the book dataset from Kaggle (instructions provided in the notebooks)

4. **Run the Pipeline**:
   Execute notebooks in order:

   1. [`data-exploration.ipynb`](data-exploration.ipynb)
   2. [`text-classification.ipynb`](text-classification.ipynb)
   3. [`sentiment-analysis.ipynb`](sentiment-analysis.ipynb)
   4. [`vector-search.ipynb`](vector-search.ipynb)

5. **Launch Web Application**:
   ```bash
   python gradio-dashboard.py
   ```

## Features

- **Natural Language Search**: Find books using descriptive queries
- **Emotion-Based Filtering**: Sort books by emotional content
- **Genre Classification**: Filter between fiction and non-fiction
- **Semantic Similarity**: Discover books similar to your preferences
- **Interactive Web Interface**: User-friendly Gradio dashboard
- **Comprehensive Data Pipeline**: From raw data to deployed application

This project demonstrates practical applications of modern NLP techniques, vector databases, and web application development for building intelligent recommendation systems.
