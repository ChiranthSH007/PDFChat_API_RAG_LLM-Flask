# PDFChat API using RAG_LLM_Flask_Langchain

## Project Description

### Overview:

The objective of this project is to develop an API using Flask that enables users to interact with PDF documents through natural language queries. Leveraging Large Language Models (LLMs), including 'Flacon,' 'Snoozy 13B,' 'Mistral 7B,' and 'Nous Hermes Llama 2 13B,' the system will process user queries and provide specific answers based on the content of the PDF. The project will also utilize Vectordb to enhance the efficiency of various processes.

In addition to the aforementioned components, this project will incorporate the LangChain library, adding further sophistication to the prompt templates, regular expression splitting, vector index creation, and embedding processes. LangChain will play a pivotal role in enhancing the natural language processing capabilities of the system, making it even more versatile and adaptive.

### Key Features:

1. Flask API:
   The project will employ Flask, a lightweight web framework, to create a RESTful API.
   Endpoints will be designed to handle user requests, facilitating communication between the client and the server.

2. PDF Content Extraction:
   PDF content extraction will be implemented to convert PDF documents into machine-readable text.
   Libraries such as PyPDF2 or pdfminer will be used to extract text and metadata from PDF files.

3. LangChain Integration:
   LangChain, a powerful language processing library, will dynamically manage prompt templates and enhance the construction of queries.
   It will facilitate regex splitting, breaking down complex queries to improve understanding and responsiveness.

4. VectorDB Integration:
   Vectordb will be utilized to store and manage vectorized representations of PDF content.
   This integration aims to optimize the retrieval and comparison of document vectors during user queries.

5. Hugging Face Embedding:
   Hugging Face embeddings will be employed to enhance the linguistic context of LLMs, improving their ability to generate contextually relevant responses.

6. Large Language Models (LLMs):
   The project will support four different LLMs: 'Flacon,' 'Snoozy 13B,' 'Mistral 7B,' and 'Nous Hermes Llama 2 13B.'
   These LLMs will be employed for natural language understanding and generation, allowing the system to comprehend user queries and generate relevant responses.

7. Query Processing:
   User queries will be preprocessed to identify key terms and intent.
   The selected LLM will analyze the query and generate a response based on its understanding of the content.

8. Answer Retrieval:
   The API will use vectorized representations of both the user query and PDF content to identify the most relevant answers.
   Vectordb will aid in efficiently retrieving and comparing document vectors to provide accurate responses.

### Project Workflow:

1. Data Collection and Preprocessing: Collect and preprocess a diverse dataset of vegetable images, properly labeling and augmenting the data.
2. Model Building: Utilize the VGG16 architecture, adapt it for vegetable classification, and train it using the preprocessed dataset.
3. Web Application Development: Create a user-friendly web application using HTML, CSS, and Flask. Implement features for image uploads, model inference, and results presentation.
4. CI/CD with GitHub Actions: Set up CI/CD pipelines on GitHub to automate testing and deployment of code changes.
5. Docker Containerization: Containerize the web application using Docker for ease of deployment and scalability.
6. Deployment on AWS: Host the web application on an AWS EC2 instance to make it accessible to users.

### Expected Outcomes:

1. User Input with LangChain Templates:
   Users provide queries, and LangChain templates dynamically structure the prompts to capture a broader range of language patterns.

2. LangChain for Query Parsing:
   LangChain utilizes regex splitting to break down complex queries into components, improving the system's understanding of user intent.

3. Hugging Face Embedding Integration:
   LangChain seamlessly integrates Hugging Face embeddings into the query processing pipeline, enhancing the linguistic context for LLMs.

4. Vectorized Indexing with Vectordb and LangChain:
   LangChain contributes to the creation and maintenance of a vector index, optimizing the storage and retrieval of vectorized PDF content representations with Vectordb.

5. LLM Processing with Enriched Embeddings:
   LLMs, including 'Flacon,' 'Snoozy 13B,' 'Mistral 7B,' and 'Nous Hermes Llama 2 13B,' process queries using LangChain-enhanced embeddings, improving response relevance.

6. Response Generation:
   The API, powered by LangChain, LLMs, Hugging Face embeddings, and Vectordb, returns enriched and contextually relevant responses to the user.

# How to run?

### STEPS:

Clone the repository

```bash
https://github.com/ChiranthSH007/PDFChat_API_RAG_LLM-Flask
```

### STEP 01- Create a python environment after opening the repository

```bash
python -m venv pdfchatenv
```

```bash
source pdfchatenv/Scripts/activate
```

### STEP 02- install the requirements

```bash
pip install -r requirements.txt
```

```bash
# Finally run the following command
python app.py
```

Now,

```bash
open up you local host and port
```

### API Endpoints

```bash
http://127.0.0.1:8080/loadmodel
```

- Parameters:
  model:Falcon
  filepath:C:/Users/win10/Desktop/Materials/Deep Learning/Deep Learning Notes.pdf
  chunksize:500
  overlap:50

```bash
http://127.0.0.1:8080/params
```

- Parameters:
  topk:2
  temp:0.7
  threads:64
  ragoff:False
  chunksize:500
  maxtoken:50

```bash
http://127.0.0.1:8080/chat
```

- Parameters:
  querytext:who is the author of this book
  ragoff:False
