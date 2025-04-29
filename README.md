# Real-Time-ArXiv-Paper-Recommendation-System

## Overview
The **Real Time ArXiv Paper Recommendation System** is a cutting-edge tool designed to streamline literature discovery within the arXiv repository. It employs semantic search powered by all-MiniLM-L6-v2 and efficient vector search with FAISS to deliver contextually relevant paper recommendations. Built with Streamlit, it offers an intuitive interface for researchers, students, and academics to explore preprints effectively.

## Demo Video
[ArXiv_Paper_Recommendation_System.webm](https://github.com/user-attachments/assets/3a1b3e1a-4cc6-4b55-b302-fd85ef5e69b4)


## Features

- **Semantic Recommendations**: Captures contextual meaning using ``all-MiniLM-L6-v2`` embeddings, overcoming keyword limitations.



- **Dynamic Dataset Updates**: Integrates new papers from the arXiv API into arxiv_metadata.csv.



- **Interactive Interface**: Streamlit-based UI with customizable filters (e.g., categories, years, authors).



- **Scalable Search**: Utilizes FAISS for efficient large-scale vector search.



- **Hardware Support**: Compatible with GPU (e.g., MPS on Apple Silicon) and CPU for embedding tasks.

## Tech Stack
- **Python**: Core programming language.



- **Streamlit**: Framework for the interactive web interface.



- **Sentence-Transformers**: Library for ``all-MiniLM-L6-v2`` embeddings.



- **FAISS**: Library for vector similarity search (e.g., ``IndexFlatL2``).



- **arXiv API**: Source for fetching paper metadata.



- **Pandas**: For data manipulation and CSV handling.



- **NumPy**: For numerical computations in embeddings.


## Dataset Description
The dataset is primarily based on ``arxiv_metadata.csv``, a file containing metadata of arXiv papers fetched via the [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv). It includes fields such as paper titles, abstracts, authors, categories, and publication dates, updated dynamically with user-specified queries (e.g., "machine learning"). The dataset supports up to 200 results per API call, with duplicates removed using paper IDs, and is preprocessed for embedding generation with ``all-MiniLM-L6-v2``.


## Example Workflow
- **Update Dataset**: Enter a query like "machine learning" and fetch 50 new papers to append to arxiv_metadata.csv.



- **Generate Embeddings**: Process the dataset with all-MiniLM-L6-v2 to create semantic vectors, accelerated by GPU if available.



- **Search Papers**: Input a research interest (e.g., "neural networks") in the Streamlit app, apply filters, and receive 5 relevant papers.



- **Review Results**: Explore titles, abstracts, and arXiv links, refining the query as needed.


## Steps to run the AI-Powered Multi-Modal Document Inteligence in your system
- ### Clone the Repository
Open a terminal and run the following command to clone the repository:

```
git clone https://github.com/AmaanSyed110/Real-Time-ArXiv-Paper-Recommendation-System.git
```
- ### Set Up a Virtual Environment
It is recommended to use a virtual environment for managing dependencies:

```
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```
- ### Install Dependencies
Install the required packages listed in the ```requirements.txt``` file
```
pip install -r requirements.txt
```

- ### Run the Application
Launch the Streamlit app by running the following command:
```
streamlit run main_app.py
```


## Contributions
Contributions are welcome! Please fork the repository and create a pull request for any feature enhancements or bug fixes.


