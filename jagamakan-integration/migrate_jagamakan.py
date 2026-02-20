import os
import time  # <-- Added this to handle the API pauses
import boto3
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client

load_dotenv()

# 1. Plug in your Cloud and AWS keys!
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # Make sure this is set in your .env file
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") 

# Your AWS IAM credentials
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
bucket_name = 'langchain-rag-storage'
file_key = "latest-01.Buku-MDG-2020_12Mac2024.pdf" # Make sure this matches your S3 object key

print("Connecting to Supabase...")
supabase = create_client(supabase_url, supabase_key)

# 2. Download the document from AWS S3
print("Fetching the Dietary Guidelines from S3...")
s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
s3.download_file(bucket_name, file_key, "temp_guidelines.pdf")

# 3. Load and split the downloaded PDF
print("Reading and chunking the document...")
loader = PyPDFLoader("temp_guidelines.pdf") 
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# 4. Generate embeddings and push to Supabase (Batched for rate limits)
print("Embedding and uploading to the Supabase cloud...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Initialize the vector store connection FIRST
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

# Set the batch size to stay safely under the 100 requests/min limit
batch_size = 80 
total_chunks = len(chunks)
print(f"Total document chunks to embed: {total_chunks}")

# Loop through the chunks and upload them in batches
for i in range(800, total_chunks, batch_size): # <-- Starting at 800 to skip the first 800 chunks that were already uploaded in the previous run
    batch = chunks[i : i + batch_size]
    print(f"Processing batch {i} to {i + len(batch)}...")
    
    # Send this specific batch to be embedded and saved in Supabase
    vector_store.add_documents(batch)
    
    # If there are still more chunks left, pause for 65 seconds
    if i + batch_size < total_chunks:
        print("Pausing for 65 seconds to respect Gemini API free tier limits...")
        time.sleep(65)

# 5. Clean up the temporary file
os.remove("temp_guidelines.pdf")
print("✅ Successfully migrated data from S3 to Supabase!")