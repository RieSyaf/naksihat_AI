import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client

load_dotenv()

# 1. Define Request and Response Schemas
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

# 2. Initialize the Router
router = APIRouter(prefix="/api/jagamakan", tags=["JagaMakan Dietary Bot"])

# 3. Initialize Connections (This happens once when the server starts)
try:
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )
    
    # We use gemini-1.5-flash for fast, free-tier friendly responses
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 most relevant chunks
except Exception as e:
    print(f"Error initializing JagaMakan AI components: {e}")

# 4. The Actual Endpoint
@router.post("/chat", response_model=ChatResponse)
async def dietary_chat(request: ChatRequest):
    try:
        # A. Retrieve relevant context from Supabase
        docs = retriever.invoke(request.query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # B. Prompt Gemini with the context
        prompt = f"""
        You are a helpful and knowledgeable Malaysian Dietary Assistant.
        Use the following retrieved context from the 2020 Malaysian Dietary Guidelines to answer the user's question. 
        If you don't know the answer based on the context, just say that you don't know. Do not make up information.
        
        Context:
        {context}
        
        User Question: {request.query}
        
        Answer:
        """
        
        # C. Get the answer
        response = llm.invoke(prompt)
        
        return ChatResponse(answer=response.content)
        
    except Exception as e:
        import traceback
        traceback.print_exc()  # <--- This forces the terminal to print the full red error log
        raise HTTPException(status_code=500, detail=str(e))