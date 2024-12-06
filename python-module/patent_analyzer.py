import os
import gradio as gr
from datasets import load_dataset
from transformers import pipeline
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate


class PatentAnalyzer:
    def __init__(self, dataset_name="big_patent", dataset_config="g", num_samples=300, chunk_size=1024, chunk_overlap=50):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.num_samples = num_samples
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_db = self.load_or_create_vectordb()

    def load_or_create_vectordb(self):
      try:
        return FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)
      except Exception as e:
        print(f"Error loading vector database: {e}. Creating a new one.")
        dataset = load_dataset(self.dataset_name, self.dataset_config, split=f"train[:{self.num_samples}]")
        texts = dataset["description"]
        return self.create_vectordb(texts)

    def create_vectordb(self, texts):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        docs = []
        for i, text in enumerate(tqdm(texts, desc="Creating chunks")):
            splits = text_splitter.split_text(text)
            for chunk in splits:
                docs.append(Document(page_content=chunk, metadata={"source": f"patent_{i + 1}"}))

        db = FAISS.from_documents(documents=docs, embedding=self.embeddings)
        db.save_local("faiss_index")  # Save after creation
        return db

    def get_qa_chain(self):
    
        retriever = self.vector_db.as_retriever(search_type="mmr",
        search_kwargs={"k": 4, "fetch_k":9}
        )
        qa_chain = RetrievalQA.from_llm(
            llm = ChatOpenAI(model_name="gpt-4o", temperature=0, max_tokens=1000),
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )
        # qa_chain = ConversationalRetrievalChain.from_llm(
        #     llm = ChatOpenAI(model_name="gpt-4o", temperature=0, max_tokens=1000),
        #     retriever=retriever,
        #     memory=memory,
        #     return_source_documents=True
        # )
        
        sys_prompt = "Answer the question based on the context provided only. Do not add any new information."
        qa_chain.combine_documents_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate.from_template(sys_prompt)
        return qa_chain

    def generate_response(self, query, context):
        llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o", max_tokens=256)
        prompt_template = "Answer the query by using the context provided only.\
        Do not add any additional information. If the context is not relevant enough then\
        say Context not relevant enough. query: {query}\n context: {context}"
        prompt = PromptTemplate(template=prompt_template, input_variables=["query", "context"])
        return llm(prompt.format(query=query, context=context)).content

    def get_summarized_results(self, retrieval_query, generation_query, search_kwargs={"k": 3, "fetch_k": 5}):
        retriever = self.vector_db.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
        source_documents = retriever.invoke(retrieval_query)
        source_content_dict = {}
        summarized_results = ""
        source_metadata = ""
        for doc in source_documents:
            source = doc.metadata['source']
            source_content_dict[source] = source_content_dict.get(source, "") + doc.page_content
        for source, context in source_content_dict.items():
            summary = self.generate_response(query=generation_query, context=context)
            summarized_results += f"Source: {source}\n" + summary + "\n\n"
            source_metadata += f"Source: {source}\nExcerpt: {context}\n\n"
        return summarized_results, source_metadata

    def get_retrieved_docs_metadata(self, result):
        # Get metadata of retrieved documents
        retrieved_docs_metadata = ""
        if 'source_documents' in result:
            for doc in result['source_documents']:
                if hasattr(doc, 'metadata'):
                    retrieved_docs_metadata += f"Metadata: {doc.metadata}\nExcerpt: {doc.page_content}\n\n"
        return retrieved_docs_metadata
    
    def patent_summarization(self, patent_text, query):
        llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o", max_tokens=256)
    
        prompt_template = """{query}
        
        Patent text: {text}"""
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["query", "text"])
        
        response = llm(prompt.format(query=query, text=patent_text))
        return response.content
    
    def prior_art_search(self, description):
        return self.get_summarized_results(retrieval_query=description, generation_query=description)
    
    def competitive_monitoring(self, technology_area):
        generation_query = "Analyze the following text and provide: 1. Summary\n 2. Date of filing if available\n 3.Organization or company name if available."
        return self.get_summarized_results(retrieval_query=technology_area, generation_query=generation_query)

    def claim_analysis(self, claim1, claim2):
        qa_chain = self.get_qa_chain()
        result = qa_chain(
            {
                "query": f"Compare and contrast these two claims:\nClaim 1: {claim1}\nClaim 2: {claim2}"
            }
        )
        retrieved_docs_metadata = self.get_retrieved_docs_metadata(result)
        return (result["result"], retrieved_docs_metadata)

    def landscape_overview(self, cpc_code):
        return self.get_summarized_results(retrieval_query=cpc_code, generation_query="Summarize: ")
        