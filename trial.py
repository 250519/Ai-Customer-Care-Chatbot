import os
from typing import List
import gradio as gr
from PyPDF2 import PdfReader
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_pinecone import Pinecone
from pinecone import Pinecone as PineconeClient
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

class CustomerServiceBot:
    def __init__(self, pdf_path: str):
        load_dotenv()
        
        # Initialize configurations
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = "us-east-1"
        self.index_name = "safegaurd"
        self.namespace = "default"
        
        if not all([self.google_api_key, self.pinecone_api_key]):
            raise ValueError("Please set all required API keys in .env file")
            
        # Initialize Pinecone
        self.pinecone_client = PineconeClient(api_key=self.pinecone_api_key)
        
        # Initialize components
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        self.llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.google_api_key,
            temperature=0.7
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Initialize vector store by processing the PDF
        self.vector_store = self.process_pdf(pdf_path)
        
        # Initialize conversation memory with explicit output key
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.PROMPT = PromptTemplate(
            template="""You are a knowledgeable representative for Safeguard Realty, a Digital Asset Platform focusing on real estate tokenization. Use the following context and chat history to answer questions about the company, its services, team, and offerings. Be specific and use details from the context when available.

Context information is below:
{context}

Chat History:
{chat_history}

Current question: {question}

Instructions:
1. Focus on Safeguard Realty's key aspects:
   - Digital Asset Platform for Land Registry
   - Tokenization of real world assets
   - Team members and their expertise
   - Business model and revenue streams
   - Market potential and growth plans
2. If the information is in the context or previous chat, provide detailed answers
3. Reference specific data points and figures when available
4. Maintain a professional and informative tone
5. For technical questions about tokenization or blockchain, use explanations from the pitch deck
6. When discussing financials or market size, use the exact figures from the document

Answer:""",
            input_variables=["context", "chat_history", "question"]
        )

    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        text = re.sub(r'\s+', ' ', text)
        text = text.replace(' v ', 'v')
        text = text.replace(' e ', 'e')
        text = text.replace(' r ', 'r')
        text = ' '.join(text.split())
        return text

    def process_pdf(self, pdf_path: str) -> Pinecone:
        """Process PDF file with improved text extraction"""
        try:
            index = self.pinecone_client.Index(self.index_name)
            stats = index.describe_index_stats()
            
            if self.namespace in stats.get('namespaces', {}):
                print(f"Using existing document: {self.namespace}")
                return Pinecone(
                    index_name=self.index_name,
                    embedding=self.embeddings,
                    namespace=self.namespace
                )
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text_parts = []
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    cleaned_text = self.clean_text(text)
                    text_parts.append(cleaned_text)
                
                full_text = ' '.join(text_parts)
                print(f"Extracted and cleaned text length: {len(full_text)} characters")
                
                doc = Document(page_content=full_text, metadata={"source": pdf_path})
            
            splits = self.text_splitter.split_documents([doc])
            print(f"Created {len(splits)} text chunks")
            
            if splits:
                print(f"First chunk preview (cleaned):\n{splits[0].page_content[:300]}...")
            
            vector_store = Pinecone.from_documents(
                documents=splits,
                embedding=self.embeddings,
                index_name=self.index_name,
                namespace=self.namespace
            )
            
            print(f"Successfully processed {len(splits)} sections from PDF")
            return vector_store
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    def get_response(self, message: str, history: List) -> str:
        """Generate response with chat memory"""
        try:
            print(f"\nProcessing question: {message}")
            
            # Convert gradio history to chat memory format
            if history:
                for human, ai in history:
                    self.memory.save_context({"human": human}, {"answer": ai})
            
            # Create retriever
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 5},
                namespace=self.namespace
            )
            
            # Create QA chain with memory
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": self.PROMPT},
                return_source_documents=True,
                verbose=True
            )
            
            # Get response
            response = qa_chain({"question": message})
            print(f"Response generated: {response['answer'][:200]}...")
            return response["answer"]
            
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."


def create_interface(pdf_path: str) -> gr.Blocks:
    """Create Gradio interface"""
    bot = CustomerServiceBot(pdf_path)
    
    with gr.Blocks(title="Customer Service AI Assistant") as interface:
        gr.Markdown("# Customer Service AI Assistant")
        
        chatbot_interface = gr.ChatInterface(
            bot.get_response,
            chatbot=gr.Chatbot(height=400),
            textbox=gr.Textbox(
                placeholder="Ask me anything about our products or services...",
                container=False,
                scale=7
            )
        )
    
    return interface

if __name__ == "__main__":
    PDF_PATH = "/Users/harshshivhare/Ai-Customer-Care-Chatbot/SafeguardRealty_VCPitchDeck_Upload (1).pdf"
    
    demo = create_interface(PDF_PATH)
    demo.launch(share=True)