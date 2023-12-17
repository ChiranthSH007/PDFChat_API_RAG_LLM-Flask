import contextlib
import pandas as pd
import time
import io

# import libs
from tqdm import tqdm
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceEmbeddings


class RAGPDFBot:

    def __init__(self):
        self.model_path = ""
        self.file_path = ""
        self.user_input = ""
        self.model = ""

    def get_model(self, model, chunk_size: int = 10000):
        # set the local paths of the models
        self.model = model
        if self.model == "Falcon":
            self.model_path = "E:/Programming/Machine Learning/LLM/GPT4All_llms/gpt4all-falcon-q4_0.gguf"
        elif self.model == "Snoozy 13B":
            self.model_path = "E:/Programming/Machine Learning/LLM/GPT4All_llms/gpt4all-13b-snoozy-q4_0.gguf"
        elif self.model == "Mistral 7B":
            self.model_path = "E:/Programming/Machine Learning/LLM/GPT4All_llms/mistral-7b-openorca.Q4_0.gguf"
        elif self.model == "Nous Hermes Llama 2 13B":
            self.model_path = "E:/Programming/Machine Learning/LLM/GPT4All_llms/nous-hermes-llama2-13b.Q4_0.gguf"

    def build_vectordb(self, chunk_size, overlap, file_path):
        # Loading the PDF
        loader = PyPDFLoader(file_path)
        # Splitting the text in the PDF Recursively
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap)
        # Create the vectordb with required embedding
        self.index = VectorstoreIndexCreator(embedding=HuggingFaceEmbeddings(
        ), text_splitter=text_splitter).from_loaders([loader])

    def load_model(self, n_threads, max_tokens, repeat_penalty, n_batch, top_k, temp):
        # load the model with the required hyper-parameters
        callbacks = [StreamingStdOutCallbackHandler()]

        self.llm = GPT4All(model=self.model_path, callbacks=callbacks, verbose=False,
                           n_threads=n_threads, n_predict=max_tokens, repeat_penalty=repeat_penalty, n_batch=n_batch, top_k=top_k, temp=temp)

    def retrieval(self, user_input, top_k, context_verbosity=False, rag_off=False):
        self.user_input = user_input
        self.context_verbosity = context_verbosity
        result = self.index.vectorstore.similarity_search(
            self.user_input, k=top_k)
        context = "\n".join([document.page_content for document in result])

        if self.context_verbosity:
            print(f"Retrieving information related to your question...")
            print(
                f"Found this content which is most similar to your question:{context}")
        # Creating the Prompt Template for the model
        if rag_off:
            template = """Question: {question}
            Answer: This is the response:
            """
            self.prompt = PromptTemplate(
                template=template, input_variables=["question"])
        else:
            template = """Dont't just repeat  the following context, use it in conbination with your knowledge to improve your answer to the question: {context}
            Question: {question}
            """
            self.prompt = PromptTemplate(template=template, input_variables=[
                                         "context", "question"]).partial(context=context)

    def inference(self):
        if self.context_verbosity:
            print(f"Your Query: {self.prompt}")
        # Using Langchian to prompt the model
        llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)
        print(f"Processing the information...\n")
        response = llm_chain.run(self.user_input)

        return response
    # 'Flacon','Snoozy 13B','Mistral 7B','Nous Hermes Llama 2 13B'
