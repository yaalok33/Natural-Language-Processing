# Natural Language Processing (NLP) Experiments

A comprehensive collection of **32 Natural Language Processing (NLP)
practical experiments** implemented using Python and popular NLP,
machine learning, and deep learning libraries.

The repository covers the complete NLP workflow, from basic text
preprocessing and linguistic analysis to text representation, semantic
similarity, classification, information retrieval, sequence modeling,
and transformer-based applications.

The experiments are organized into three learning sections:

-   **Experiments 01--07:** NLP fundamentals and text preprocessing
-   **Experiments 08--23:** Text representation, similarity, machine
    learning, and sequence models
-   **Experiments 24--32:** Advanced NLP using transformers and
    document-processing techniques

------------------------------------------------------------------------

## Table of Contents

-   [Experiments Included](#-experiments-included)
    -   [Section 1: NLP Fundamentals
        (01--07)](#section-1-nlp-fundamentals-01-07)
    -   [Section 2: NLP Techniques and Applications
        (08--23)](#section-2-nlp-techniques-and-applications-08-23)
    -   [Section 3: Advanced NLP
        (24--32)](#section-3-advanced-nlp-24-32)
-   [Technologies Used](#-technologies-used)
-   [Installation and Setup](#-installation-and-setup)
-   [Running the Notebooks](#-running-the-notebooks)
-   [Repository Structure](#-repository-structure)
-   [Learning Objectives](#-learning-objectives)
-   [Notes](#-notes)
-   [Course Outcome](#-course-outcome)
-   [Author](#-author)
-   [License](#-license)

------------------------------------------------------------------------

## 📚 Experiments Included

### Section 1: NLP Fundamentals (01--07)

This section introduces the foundational techniques used to process and
analyze natural language text.

  -----------------------------------------------------------------------
  No.                     Experiment              Description
  ----------------------- ----------------------- -----------------------
  01                      Tokenization            Splitting text into
                                                  sentences and words
                                                  using NLTK and spaCy.

  02                      Stemming and            Finding word stems and
                          Lemmatization           meaningful dictionary
                                                  base forms using Porter
                                                  Stemmer and WordNet
                                                  Lemmatizer.

  03                      Stop-word Removal       Identifying and
                                                  removing common words
                                                  that may contribute
                                                  limited information to
                                                  a particular NLP task.

  04                      Part-of-Speech (POS)    Assigning grammatical
                          Tagging                 categories such as
                                                  nouns, verbs,
                                                  adjectives, and adverbs
                                                  to words.

  05                      Parsing and Chunking    Performing syntactic
                                                  analysis using
                                                  regular-expression
                                                  chunking, noun phrase
                                                  chunking, and
                                                  dependency parsing.

  06                      Named Entity            Detecting and
                          Recognition (NER)       classifying entities
                                                  such as people,
                                                  organizations,
                                                  locations, dates, and
                                                  monetary values using
                                                  spaCy.

  07                      Additional NLP          Reserved for an
                          Fundamentals            additional foundational
                                                  practical, if included
                                                  in the repository.
  -----------------------------------------------------------------------

#### Concepts Covered

-   Sentence and word tokenization
-   Stemming and lemmatization
-   Stop-word identification and removal
-   Part-of-Speech tagging
-   Regular-expression-based chunking
-   Noun phrase chunking
-   Dependency parsing
-   Named Entity Recognition

**Main libraries:** Python, NLTK, spaCy, and Jupyter Notebook.

### Section 2: NLP Techniques and Applications (08--23)

This section focuses on converting text into numerical representations,
measuring semantic similarity, building machine learning models, and
extracting useful information from documents.

  -----------------------------------------------------------------------
  No.                     Experiment              Description
  ----------------------- ----------------------- -----------------------
  08                      Bag of Words (BoW)      Representing text
                                                  documents using
                                                  word-frequency vectors.

  09                      TF-IDF Implementation   Implementing Term
                                                  Frequency--Inverse
                                                  Document Frequency and
                                                  comparing it with Bag
                                                  of Words.

  10                      N-Gram Model            Generating unigram,
                                                  bigram, and trigram
                                                  models from a text
                                                  corpus.

  11                      Cosine Similarity       Computing similarity
                                                  between text documents
                                                  using vector
                                                  representations.

  12                      Word2Vec Embeddings     Creating word
                                                  embeddings using Gensim
                                                  on a custom corpus.

  13                      GloVe Embeddings        Loading pretrained
                                                  GloVe embeddings and
                                                  representing words as
                                                  vectors.

  14                      Word Mover's Distance   Measuring semantic
                          (WMD)                   distance between
                                                  documents using word
                                                  embeddings.

  15                      Text Classification     Classifying text using
                                                  machine learning
                                                  algorithms such as
                                                  Naive Bayes and Support
                                                  Vector Machines (SVM)
                                                  with TF-IDF features.

  16                      Sentiment Analysis      Identifying sentiment
                                                  in text using TextBlob
                                                  and VADER.

  17                      Topic Modeling using    Discovering latent
                          LDA                     topics in a document
                                                  collection using Latent
                                                  Dirichlet Allocation.

  18                      Topic Modeling using    Extracting topics using
                          LSA                     Latent Semantic
                                                  Analysis.

  19                      Opinion Mining          Analyzing opinions
                                                  expressed in product
                                                  and service reviews.

  20                      Information Extraction  Extracting useful
                                                  information from
                                                  structured and
                                                  unstructured documents.

  21                      Information Retrieval   Building a document
                                                  retrieval system and
                                                  ranking results using
                                                  TF-IDF.

  22                      Sequence Classification Performing sequence
                                                  classification using
                                                  RNN and LSTM models
                                                  with Keras and
                                                  TensorFlow.

  23                      Attention Mechanism     Implementing a basic
                                                  custom attention
                                                  mechanism for sequence
                                                  modeling.
  -----------------------------------------------------------------------

#### Concepts Covered

-   Text vectorization and feature extraction
-   Bag of Words and TF-IDF
-   N-gram language models
-   Cosine similarity and semantic distance
-   Word embeddings using Word2Vec and GloVe
-   Text classification
-   Sentiment analysis and opinion mining
-   Topic modeling using LDA and LSA
-   Information extraction and information retrieval
-   Recurrent Neural Networks (RNN)
-   Long Short-Term Memory (LSTM)
-   Attention mechanisms

**Main libraries:** Python, NLTK, spaCy, NumPy, pandas, scikit-learn,
Gensim, TextBlob, VADER Sentiment, TensorFlow, Keras, and Matplotlib.

### Section 3: Advanced NLP (24--32)

This section explores transformer-based models and advanced
document-processing techniques using pretrained language models and
machine learning.

  -----------------------------------------------------------------------
  No.                     Experiment              Description
  ----------------------- ----------------------- -----------------------
  24                      Fine-Tuning BERT for    Fine-tuning a
                          Text Classification     pretrained BERT model
                                                  for text classification
                                                  using Hugging Face
                                                  Transformers.

  25                      Sentence Embeddings     Generating sentence
                          using BERT and GPT-2    representations using
                                                  transformer-based
                                                  language models.

  26                      Text Generation using   Generating text using
                          GPT-2                   GPT-2 with a custom
                                                  dataset or prompt.

  27                      Named Entity            Identifying entities
                          Recognition using       such as people,
                          Transformers            locations, and
                                                  organizations using
                                                  transformer-based
                                                  models.

  28                      Machine Translation     Translating text using
                          using MarianMT or T5    pretrained Hugging Face
                                                  models such as MarianMT
                                                  or T5.

  29                      Text Summarization      Generating concise
                          using BART or T5        summaries from longer
                                                  text documents using
                                                  pretrained transformer
                                                  models.

  30                      Text Summarization:     Exploring an additional
                          Alternative             implementation or
                          Implementation          configuration of
                                                  transformer-based text
                                                  summarization.

  31                      Automatic Document      Classifying documents
                          Classification and      into categories and
                          Separation              organizing them using
                                                  machine learning
                                                  techniques.

  32                      Hybrid Probabilistic    Combining probabilistic
                          and FSM-Based Sequence  methods with Finite
                          Modeling                State Machines (FSM)
                                                  for sequence modeling
                                                  and document-boundary
                                                  detection.
  -----------------------------------------------------------------------

#### Concepts Covered

-   Transformer architecture and pretrained language models
-   BERT fine-tuning
-   Sentence embeddings
-   GPT-2 text generation
-   Transformer-based Named Entity Recognition
-   Machine translation
-   Abstractive text summarization
-   Document classification and separation
-   Probabilistic sequence modeling
-   Finite State Machines (FSM)

**Main libraries:** Python, Hugging Face Transformers, BERT, GPT-2,
BART, T5, MarianMT, PyTorch, TensorFlow/Keras, scikit-learn, NumPy, and
pandas.

------------------------------------------------------------------------

## 🛠️ Technologies Used

The repository uses the following tools and libraries across its
experiments:

  -----------------------------------------------------------------------
  Category                            Technologies
  ----------------------------------- -----------------------------------
  Programming Language                Python 3

  Development Environment             Jupyter Notebook, JupyterLab,
                                      Google Colab, VS Code

  NLP Libraries                       NLTK, spaCy

  Data Processing                     NumPy, pandas

  Machine Learning                    scikit-learn

  Word Embeddings                     Gensim, Word2Vec, GloVe

  Sentiment Analysis                  TextBlob, VADER Sentiment

  Visualization                       Matplotlib

  Deep Learning                       TensorFlow, Keras, PyTorch

  Transformer Models                  Hugging Face Transformers, BERT,
                                      GPT-2, BART, T5, MarianMT
  -----------------------------------------------------------------------

Not every experiment requires every library. Install additional packages
only when needed by a particular notebook.

------------------------------------------------------------------------

## 🚀 Installation and Setup

### 1. Install Python

Install Python 3 on your system. It is recommended to use a virtual
environment so that project dependencies remain isolated.

### 2. Clone the Repository

``` bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Replace the placeholders with your actual GitHub repository URL and
folder name.

### 3. Create and Activate a Virtual Environment (Recommended)

**Windows:**

``` bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Required Libraries

Install the packages used across the experiments:

``` bash
pip install jupyter numpy pandas scikit-learn nltk spacy gensim textblob vaderSentiment matplotlib tensorflow torch transformers sentencepiece
```

Some packages, pretrained models, and datasets may be needed only for
specific notebooks. If an experiment reports a missing dependency,
install the package mentioned in that notebook.

### 5. Download the spaCy English Language Model

``` bash
python -m spacy download en_core_web_sm
```

### 6. Download Required NLTK Resources

Some notebooks download their required NLTK data automatically. If
resources are missing, run:

``` python
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

Depending on the NLTK version and notebook, additional resources may be
required.

------------------------------------------------------------------------

## ▶️ Running the Notebooks

The experiments are implemented as Jupyter Notebook (`.ipynb`) files.

You can run them using:

-   Jupyter Notebook
-   JupyterLab
-   Google Colab
-   Visual Studio Code with the Jupyter extension

### Start Jupyter Notebook

``` bash
jupyter notebook
```

A browser window will open. Navigate to the required experiment and open
its notebook.

Run the notebook cells sequentially, from top to bottom.

For example:

``` text
01_Tokenization/tokenization.ipynb
```

For the other experiments, open the corresponding notebook file listed
in the repository.

------------------------------------------------------------------------

## 📂 Repository Structure

The following is a suggested structure for organizing all experiments.
Keep the actual filenames and folder names in your repository consistent
with this structure, or update this section to match them.

``` text
NLP-Experiments/
│
├── 01_Tokenization/
│   └── tokenization.ipynb
│
├── 02_Stemming_Lemmatization/
│   └── stemming_lemmatization.ipynb
│
├── 03_Stopword_Removal/
│   └── stopword_removal.ipynb
│
├── 04_POS_Tagging/
│   └── pos_tagging.ipynb
│
├── 05_Parsing_Chunking/
│   └── parsing_chunking.ipynb
│
├── 06_Named_Entity_Recognition/
│   └── ner.ipynb
│
├── 07_Additional_NLP_Fundamentals/
│   └── additional_nlp_fundamentals.ipynb
│
├── 08_Bag_of_Words/
│   └── bag_of_words.ipynb
│
├── 09_TF_IDF/
│   └── tf_idf.ipynb
│
├── 10_N_Gram_Model/
│   └── n_gram_model.ipynb
│
├── 11_Cosine_Similarity/
│   └── cosine_similarity.ipynb
│
├── 12_Word2Vec_Embeddings/
│   └── word2vec.ipynb
│
├── 13_GloVe_Embeddings/
│   └── glove_embeddings.ipynb
│
├── 14_Word_Movers_Distance/
│   └── word_movers_distance.ipynb
│
├── 15_Text_Classification/
│   └── text_classification.ipynb
│
├── 16_Sentiment_Analysis/
│   └── sentiment_analysis.ipynb
│
├── 17_Topic_Modeling_LDA/
│   └── topic_modeling_lda.ipynb
│
├── 18_Topic_Modeling_LSA/
│   └── topic_modeling_lsa.ipynb
│
├── 19_Opinion_Mining/
│   └── opinion_mining.ipynb
│
├── 20_Information_Extraction/
│   └── information_extraction.ipynb
│
├── 21_Information_Retrieval/
│   └── information_retrieval.ipynb
│
├── 22_Sequence_Classification_RNN_LSTM/
│   └── sequence_classification.ipynb
│
├── 23_Attention_Mechanism/
│   └── attention_mechanism.ipynb
│
├── 24_BERT_Text_Classification/
│   └── bert_text_classification.ipynb
│
├── 25_Sentence_Embeddings/
│   └── sentence_embeddings.ipynb
│
├── 26_GPT2_Text_Generation/
│   └── gpt2_text_generation.ipynb
│
├── 27_Transformer_NER/
│   └── transformer_ner.ipynb
│
├── 28_Machine_Translation/
│   └── machine_translation.ipynb
│
├── 29_Text_Summarization/
│   └── text_summarization.ipynb
│
├── 30_Advanced_Text_Summarization/
│   └── advanced_text_summarization.ipynb
│
├── 31_Document_Classification/
│   └── document_classification.ipynb
│
├── 32_Hybrid_Probabilistic_FSM_Modeling/
│   └── hybrid_probabilistic_fsm.ipynb
│
└── README.md
```

**Important:** The structure above is an organizational example.
Experiments 07 and 30 should be retained only if those practicals exist
in your repository. Replace illustrative notebook filenames with the
exact filenames you use.

------------------------------------------------------------------------

## 🎯 Learning Objectives

After completing these experiments, you should be able to:

1.  Understand the fundamentals of Natural Language Processing.
2.  Preprocess text using tokenization, stemming, lemmatization, and
    stop-word removal.
3.  Perform POS tagging, parsing, chunking, and Named Entity
    Recognition.
4.  Convert text into numerical features using BoW, TF-IDF, and N-grams.
5.  Understand word embeddings and semantic similarity.
6.  Build text classification and sentiment analysis systems.
7.  Explore topic modeling using LDA and LSA.
8.  Perform opinion mining, information extraction, and information
    retrieval.
9.  Understand sequence modeling with RNNs, LSTMs, and attention.
10. Use transformer models such as BERT, GPT-2, BART, and T5.
11. Apply pretrained models to text generation, translation,
    summarization, and entity recognition.
12. Explore document classification and hybrid sequence-modeling
    approaches.

------------------------------------------------------------------------

## 📝 Notes

-   Run notebook cells in the correct order.
-   Check each notebook for its specific dependencies and setup
    instructions.
-   Update dataset paths according to your local environment.
-   Some experiments require datasets that may need to be downloaded
    separately.
-   Word embedding experiments may require pretrained GloVe vectors or
    other embedding files.
-   Transformer experiments may download pretrained model weights from
    Hugging Face on first execution.
-   An internet connection may be required to download datasets,
    language models, and pretrained weights.
-   Large transformer and deep learning models may require substantial
    RAM or a GPU.
-   Results may vary depending on the dataset, library version, model
    configuration, and random seed.
-   The repository is intended for educational and practical learning.

------------------------------------------------------------------------

## 🎓 Course Outcome

These experiments provide hands-on experience with fundamental and
advanced Natural Language Processing techniques. They demonstrate how
raw text can be cleaned, transformed into numerical representations,
analyzed for linguistic and semantic information, and used to build
machine learning and deep learning applications.

By progressing from basic preprocessing to transformer-based systems,
the repository builds a practical understanding of the NLP pipeline and
its real-world applications.

------------------------------------------------------------------------

## 👨‍💻 Author

**Alok Yadav**

GitHub: [AlokYadav1251](https://github.com/AlokYadav1251)

------------------------------------------------------------------------

## 📄 License

This project is intended for educational and learning purposes.

If you plan to distribute the repository or allow others to reuse its
code, add an appropriate open-source license, such as the MIT License,
after confirming that it is suitable for your project.
