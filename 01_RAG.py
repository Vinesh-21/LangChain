import os

# Import Gemini chat model and embedding model
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# 1. Loader: To load raw text files
from langchain_community.document_loaders import TextLoader

# 2. Splitter: To split long text into smaller chunks
from langchain.text_splitter import CharacterTextSplitter

# 3. Vector Store: To store embeddings in Chroma (local database)
from langchain_community.vectorstores import Chroma

# Load environment variables (e.g., GOOGLE_API_KEY from .env file)
from dotenv import load_dotenv
load_dotenv()

'''
Optional LangChain utilities (not used here yet):
- RunnableLambda, RunnableParallel: Build custom chains
- StrOutputParser: Parse LLM outputs
- ChatPromptTemplate: Define prompt templates
'''

# Get current script directory path
curdir = os.path.dirname(os.path.abspath(__file__))

# Path to the text file we want to load
file_path = os.path.join(curdir, "documents", "lord_of_the_rings.txt")

# Path where Chroma DB will be persisted locally
persistent_directory = os.path.join(curdir, "db", "chroma_db")

# If vector store does NOT exist → create it
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Check if the input text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File Not Found: {file_path}")

    # ------- 1 Load the Documents -------
    loader = TextLoader(file_path)       # Load the file
    documents = loader.load()            # Returns a list of Document objects

    # ------- 2 Split the text into chunks -------
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,   # Each chunk max 1000 characters
        chunk_overlap=20   # 20 characters overlap between chunks
    )
    docs = text_splitter.split_documents(documents)  # Split into smaller docs

    # ------- 3 Create the Embeddings -------
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # ------- 4 Create and store the documents in the vector store -------
    # Stores embeddings + metadata into Chroma and persists to disk
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)

# If vector store already exists → load it instead of rebuilding
else:
    print("Persistent directory exists. Loading vector store...")
    # db = Chroma(persist_directory=persistent_directory, embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"))

# Create the Gemini chat model (used for Q&A later)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
