#   RAG-based Cyber Forensics Investigation Tool  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

##  What is RAG?

Retrieval-Augmented Generation (RAG) combines information retrieval with text generation to provide accurate, factual, and contextually relevant responses. It retrieves information from a knowledge base and uses a language model to generate answers, enabling language models to handle complex questions and access domain-specific knowledge.

This project implements a RAG system for cyber forensics investigations using LangChain, Hugging Face models, and FAISS for efficient retrieval and question answering over a provided knowledge base. The system processes a text-based scenario, splits it into manageable chunks, generates embeddings, stores them in a vector store, and uses a language model to answer user queries based on the retrieved information.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1j1mgpfdpJp0s1eQn7JzJr3lhhB8tjXLn?usp=sharing)

##  Background Story

This project uses a futuristic cyberpunk scenario to simulate a cybercrime investigation. Detective Y investigates a ransomware attack on a robotics engineer by "The Serpent," who employs advanced techniques to encrypt and steal research data. This scenario serves as the knowledge base for the RAG system.

##  Questions to be Asked

The RAG system answers questions based on the provided cyber forensics scenario. Examples:

**In-Text Questions:**

1.  What type of cyberattack did Detective Y investigate?
2.  What was the victim's profession?
3.  Where was the remote server located that led to the perpetrator's arrest?

**Out-of-Text Questions (Answers not in the text):**

1.  What specific encryption algorithm did The Serpent use?
2.  What was the name of the university where the security breach occurred?
3.  Did Detective Y's team collaborate with external experts?

##  Technical Description

The system works as follows:

1.  **Environment Setup:** Installs necessary Python libraries including `langchain`, `langchain-huggingface`, `faiss-cpu`, and `huggingface_hub`.
2.  **Scenario Definition:** Defines the cyberpunk case study as a string.
3.  **Text Splitting:** Uses `RecursiveCharacterTextSplitter` to divide the text into chunks, controlled by parameters like `chunk_size`, `chunk_overlap`, and `separators`.
4.  **Embeddings:** Generates text embeddings using the Hugging Face Inference API.
5.  **Vector Store:** FAISS is used to store and retrieve text embeddings.
6.  **Retrieval QA Chain:** LangChain's `RetrievalQA` chain combines the vector store with a language model. The chain retrieves relevant text chunks based on a user's query and generates an answer.
7.  **Language Model:** Uses the Hugging Face Inference API with a specified model (`mistralai/Mistral-7B-Instruct-v0.1`) for generating responses.
8.  **Query Processing:** The system takes user queries as input, retrieves relevant information from the vector store, and generates answers using the language model.

This setup enables the RAG system to answer questions related to the cyber forensics scenario.

##  Dependencies

To run this project, ensure you have the following:

* **Python:** Version 3.7 or higher
* **pip:** Python package installer
* **Hugging Face Account:** Required to use Hugging Face models and Inference API.
* **Google Colab Environment:** To execute the notebook.

**Disk space:** Google Colab provides a virtual environment, so disk space is managed within that environment.

##  How This Project Works

This project uses:

* **Hugging Face Models:** Pre-trained language models for embedding and text generation.
* **LangChain:** A framework for developing applications powered by language models.
* **FAISS:** A library for efficient similarity search and clustering of vectors.
* **Google Colab:** A cloud-based platform for running Python code.

The system works as follows:

1.  **Load and split:** The cyber forensics document is loaded and split into smaller chunks.
2.  **Embed:** Each chunk is embedded into a vector representation using a Hugging Face model.
3.  **Store:** The embeddings are stored in a FAISS index.
4.  **Query:** When a user asks a question, the question is embedded and used to search the FAISS index for relevant chunks.
5.  **Answer:** The relevant chunks are passed to a Hugging Face language model, which generates an answer.

##  Code Overview

The code is structured as follows:

* **Document loading and processing:** The cyber forensics document is loaded and split into chunks using `RecursiveCharacterTextSplitter`.
* **Embedding Generation:** Embeddings are generated using the Hugging Face Inference API.
* **Vectorstore creation:** A FAISS vectorstore is created using the embeddings.
* **RetrievalQA Chain:** A `RetrievalQA` chain is set up to handle question answering.
* **Chat Interface:** A simple chat interface is implemented to take user queries and display the RAG output.

##  Why This Approach Is Better

This RAG-based approach offers several advantages:

* **Contextualized responses:** The system provides answers grounded in the provided cyber forensics document.
* **Interactive interface:** The chat-like interaction provides a user-friendly experience.
* **Efficiency:** FAISS enables fast similarity search.
* **Cloud-based execution:** Google Colab provides a convenient environment.
* **Hugging Face Integration:** Leveraging pre-trained models from Hugging Face simplifies embedding and text generation.

##  System Workflow

The following outlines the workflow of the RAG system:

1.  **Document Loading:** Load the cyber forensics document.
2.  **Text Splitting:** Split the document into smaller chunks.
3.  **Embedding Generation:** Generate embeddings for each chunk using Hugging Face.
4.  **Vectorstore Creation:** Create a FAISS index from the embeddings.
5.  **User Query:** Receive a question from the user.
6.  **Embedding Query:** Embed the user's question.
7.  **Similarity Search:** Search the FAISS index for relevant chunks.
8.  **Answer Generation:** Use a Hugging Face language model to generate an answer based on the retrieved chunks.
9.  **Output:** Display the answer to the user.

##  System Workflow Diagram

The following flowchart illustrates the workflow of the RAG system:![Flowchart](Colab_RAG.png))

##  Setup and Usage

1.  **Create a Hugging Face Account** (if you don't have one): Go to [https://huggingface.co/](https://huggingface.co/) and sign up.
2.  **Generate a Hugging Face Access Token:** Log in, go to profile settings, find "Access Tokens," create a new token, and copy it.
3.  **Open Google Colab:** Open a new Google Colab notebook.
4.  **Install Python dependencies:** Run these commands in a Colab cell:

    ```bash
    !pip install transformers langchain langchain_community faiss-cpu huggingface_hub pypdf pymupdf -U langchain langchain-huggingface
    !pip install --upgrade langchain
    ```
5.  **Provide Hugging Face API Token:** Add a code cell to set the `HUGGINGFACEHUB_API_TOKEN` environment variable with your token:

    ```python
    import os
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'hf_your_token'  # Replace 'hf_your_token' with your actual token
    ```
6.  **Provide your knowledge base:** Add a cell and define your `document_text` (the scenario).
7.  **Run the code:** Execute the cells to interact with the RAG system.

##  Features

* **Google Colab Integration:** Easy setup and execution in a cloud-based environment.
* **Hugging Face Integration:** Utilizes pre-trained models for embedding and text generation.
* **FAISS Vectorstore:** Enables efficient similarity search.
* **Text Chunking:** Splits the input document into smaller, manageable chunks.
* **Chat Interface:** Provides a simple way to interact with the RAG system through text-based queries.

##  Contributing

Contributions are welcome! Feel free to open issues or pull requests for bug fixes, improvements, or new features.

##  License

This project is licensed under the MIT License.
