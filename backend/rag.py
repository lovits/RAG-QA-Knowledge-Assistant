import os
import requests
import re
import json
from typing import List, Optional, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, Docx2txtLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "chromadb_data")
RAG_RESOURCE_DIR = os.getenv("RAG_RESOURCE_DIR", "RagResource")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# Initialize Embeddings
try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
except Exception as e:
    print(f"Failed to load local embeddings from {EMBEDDING_MODEL_PATH}, falling back to default. Error: {e}")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def clean_text(text: str) -> str:
    """
    Clean text by removing excessive whitespace and non-printable characters.
    Satisfies 'Document Cleaning & Standardization' module requirement.
    """
    # Remove control characters (except newlines/tabs)
    # keeping \x09 (tab), \x0A (newline), \x0D (return)
    # removing other control chars
    text = "".join(ch for ch in text if ch.isprintable() or ch in ['\t', '\n', '\r'])
    
    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def ingest_documents():
    if not os.path.exists(RAG_RESOURCE_DIR):
        return {"status": "error", "message": "Resource directory not found."}
    
    documents = []
    
    # Load PDFs
    print(f"Loading PDF documents from {RAG_RESOURCE_DIR}...")
    try:
        # Using PyPDFLoader to get page numbers
        pdf_loader = DirectoryLoader(RAG_RESOURCE_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader, silent_errors=True)
        documents.extend(pdf_loader.load())
    except Exception as e:
        print(f"Error loading PDFs: {e}")

    # Load TXT
    print(f"Loading TXT documents from {RAG_RESOURCE_DIR}...")
    try:
        txt_loader = DirectoryLoader(RAG_RESOURCE_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}, silent_errors=True)
        documents.extend(txt_loader.load())
    except Exception as e:
        print(f"Error loading TXTs: {e}")
    
    # Load Markdown
    print(f"Loading Markdown documents from {RAG_RESOURCE_DIR}...")
    try:
        md_loader = DirectoryLoader(RAG_RESOURCE_DIR, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}, silent_errors=True)
        documents.extend(md_loader.load())
    except Exception as e:
        print(f"Error loading MDs: {e}")
    
    if not documents:
         return {"status": "warning", "message": "No documents found (PDF, TXT, MD)."}

    # Clean documents
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Create VectorDB
    print("Creating Vector Database...")
    # Use delete_collection if exists to avoid duplication or stale data if needed
    # but here we append.
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=CHROMA_DIR)
        
    return {"status": "success", "message": f"Ingested {len(texts)} chunks from {len(documents)} documents."}

def process_uploaded_file(file_path: str):
    """
    Process a single uploaded file and add to vector DB.
    """
    documents = []
    try:
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file_path.lower().endswith('.txt') or file_path.lower().endswith('.md'):
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
        elif file_path.lower().endswith('.docx'):
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
        elif file_path.lower().endswith('.xlsx') or file_path.lower().endswith('.xls'):
            # Simple Excel processing using pandas
            df = pd.read_excel(file_path)
            # Convert to string representation
            text_content = df.to_string()
            # Create a Document object manually since we don't have a simple loader for complex Excel
            from langchain_core.documents import Document
            documents = [Document(page_content=text_content, metadata={"source": file_path})]
        elif file_path.lower().endswith('.csv'):
             loader = CSVLoader(file_path)
             documents = loader.load()
        else:
            return {"status": "error", "message": "Unsupported file type"}
        
        if not documents:
             return {"status": "warning", "message": "Empty document."}

        # Clean
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)
            # Ensure metadata has clean filename
            doc.metadata['source'] = os.path.basename(file_path)

        # Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        
        # Add to VectorDB
        vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=CHROMA_DIR)
        
        return {"status": "success", "message": f"Processed {os.path.basename(file_path)} successfully ({len(texts)} chunks)."}
    except Exception as e:
        return {"status": "error", "message": f"Error processing file: {str(e)}"}

def call_deepseek(messages: List[dict], temperature: float = 0.3) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Error calling DeepSeek API: {str(e)}"

def extract_user_persona(text: str) -> Optional[str]:
    """
    Extract user persona or preferences from text.
    Returns None if no relevant info found.
    """
    messages = [
        {"role": "system", "content": "You are a user profile extractor. Extract any long-term user preferences, persona details, or role information (e.g., 'I am a student', 'I prefer tables') from the user's input. Return ONLY the extracted facts as a concise string. If no such information is present, return 'NONE'. Do not include the original question."},
        {"role": "user", "content": text}
    ]
    
    try:
        result = call_deepseek(messages, temperature=0.1).strip()
        if result == "NONE" or len(result) < 2:
            return None
        return result
    except:
        return None

def get_answer(question: str, history: List[dict] = [], user_persona: str = "") -> Dict[str, Any]:
    if not os.path.exists(CHROMA_DIR):
        return {"answer": "Database not initialized. Please ingest documents first.", "sources": []}
    
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    
    # Use similarity_search_with_score to filter irrelevant documents
    # Lower score = better match (Distance)
    # Threshold 260 filters "I am a graduate student" (264) but keeps "Landslide" (246)
    results = vectordb.similarity_search_with_score(question, k=3)
    
    SIMILARITY_THRESHOLD = 260
    
    docs = []
    for doc, score in results:
        if score < SIMILARITY_THRESHOLD:
            docs.append(doc)
            
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Extract sources
    sources = []
    seen_sources = set()
    for doc in docs:
        source = os.path.basename(doc.metadata.get('source', 'Unknown'))
        page = doc.metadata.get('page', 'N/A')
        # PyPDFLoader pages are 0-indexed, usually we want 1-indexed for display
        if isinstance(page, int):
            page += 1
        
        source_key = f"{source}-{page}"
        if source_key not in seen_sources:
            sources.append({"source": source, "page": page})
            seen_sources.add(source_key)

    # Prepare messages for DeepSeek
    system_prompt = "You are a helpful AI assistant. You have access to the provided context from documents. If the user asks a question about the documents, answer it using the context. If the user's input is conversational (e.g., greetings, persona setting) or a general instruction, respond naturally using your general knowledge. Always respond in English."
    
    if user_persona:
        system_prompt += f"\n\nUser Persona/Preferences (ALWAYS ADHERE TO THIS): {user_persona}"

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add history
    for msg in history:
        # Map frontend roles to API roles if necessary, usually 'user' and 'assistant'
        role = "assistant" if msg.get("role") == "bot" else "user"
        messages.append({"role": role, "content": msg.get("content", "")})
        
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
    
    answer = call_deepseek(messages)
    return {"answer": answer, "sources": sources}

def call_deepseek_stream(messages: List[dict], temperature: float = 0.3):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature,
        "stream": True
    }
    
    try:
        with requests.post(DEEPSEEK_API_URL, json=data, headers=headers, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith("data: "):
                        json_str = line_text[6:]
                        if json_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(json_str)
                            content = chunk['choices'][0]['delta'].get('content', '')
                            if content:
                                yield content
                        except:
                            pass
    except Exception as e:
        yield f"Error: {str(e)}"

def get_answer_stream(question: str, history: List[dict] = [], user_persona: str = ""):
    if not os.path.exists(CHROMA_DIR):
        yield json.dumps({"type": "error", "content": "Database not initialized. Please ingest documents first."}) + "\n"
        return
    
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    
    # Use similarity_search_with_score to filter irrelevant documents
    # Lower score = better match (Distance)
    # Threshold 260 filters "I am a graduate student" (264) but keeps "Landslide" (246)
    results = vectordb.similarity_search_with_score(question, k=3)
    
    SIMILARITY_THRESHOLD = 260
    
    docs = []
    for doc, score in results:
        if score < SIMILARITY_THRESHOLD:
            docs.append(doc)
            
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Extract sources
    sources = []
    seen_sources = set()
    for doc in docs:
        source = os.path.basename(doc.metadata.get('source', 'Unknown'))
        page = doc.metadata.get('page', 'N/A')
        if isinstance(page, int): page += 1
        
        source_key = f"{source}-{page}"
        if source_key not in seen_sources:
            sources.append({"source": source, "page": page})
            seen_sources.add(source_key)

    # Yield sources first
    yield json.dumps({"type": "sources", "content": sources}) + "\n"

    # Prepare messages
    system_prompt = "You are a helpful AI assistant. You have access to the provided context from documents. If the user asks a question about the documents, answer it using the context. If the user's input is conversational (e.g., greetings, persona setting) or a general instruction, respond naturally using your general knowledge. Always respond in English."
    
    if user_persona:
        system_prompt += f"\n\nUser Persona/Preferences (ALWAYS ADHERE TO THIS): {user_persona}"

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    for msg in history:
        role = "assistant" if msg.get("role") == "bot" else "user"
        messages.append({"role": role, "content": msg.get("content", "")})
        
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
    
    # Stream answer
    for token in call_deepseek_stream(messages):
        yield json.dumps({"type": "token", "content": token}) + "\n"

def export_cross_analysis_to_excel(analysis_text: str) -> str:
    """
    Parses the analysis text (Markdown/Text) and converts it into a well-formatted Excel file.
    Separates structured tables from narrative text.
    Returns the filename of the generated Excel file.
    """
    try:
        output_dir = "exports"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filename = f"analysis_export_{len(os.listdir(output_dir)) + 1}.xlsx"
        file_path = os.path.join(output_dir, filename)

        # Storage
        table_data = [] # List of dicts for the comparison table
        report_lines = [] # List of strings for the narrative report
        
        lines = analysis_text.strip().split('\n')
        
        # Parsing State
        in_table = False
        table_headers = []
        table_rows_buffer = []

        for line in lines:
            line_stripped = line.strip()
            
            # Detect Table Start or Row
            if line_stripped.startswith('|') and line_stripped.endswith('|'):
                if not in_table:
                    # Potential start of a table
                    # Check if it's a header row (next line should be separator)
                    # We'll just assume it's a table for now
                    in_table = True
                    table_rows_buffer.append(line_stripped)
                else:
                    table_rows_buffer.append(line_stripped)
            else:
                if in_table:
                    # End of table detected
                    in_table = False
                    # Process the buffered table
                    if len(table_rows_buffer) >= 2: # At least header and separator
                        try:
                            # 0: Header, 1: Separator (ignore), 2+: Data
                            headers = [h.strip() for h in table_rows_buffer[0].split('|') if h.strip()]
                            
                            # Check if valid header (sometimes | is used in text)
                            if len(headers) > 1:
                                for row_str in table_rows_buffer[2:]:
                                    cols = [c.strip() for c in row_str.split('|') if c.strip()]
                                    
                                    # Handle row length mismatch
                                    if len(cols) < len(headers):
                                        cols.extend([''] * (len(headers) - len(cols)))
                                    elif len(cols) > len(headers):
                                        cols = cols[:len(headers)]
                                        
                                    row_dict = {}
                                    for i, h in enumerate(headers):
                                        row_dict[h] = cols[i]
                                    table_data.append(row_dict)
                            else:
                                # Not a valid table, treat as text
                                report_lines.extend(table_rows_buffer)
                        except Exception as e:
                            print(f"Error parsing table block: {e}")
                            report_lines.extend(table_rows_buffer)
                    else:
                         report_lines.extend(table_rows_buffer)
                    
                    table_rows_buffer = []
                
                # Add non-table line to report
                report_lines.append(line)

        # Handle case where text ends inside a table
        if in_table and table_rows_buffer:
             if len(table_rows_buffer) >= 2:
                try:
                    headers = [h.strip() for h in table_rows_buffer[0].split('|') if h.strip()]
                    if len(headers) > 1:
                        for row_str in table_rows_buffer[2:]:
                            cols = [c.strip() for c in row_str.split('|') if c.strip()]
                            if len(cols) < len(headers):
                                cols.extend([''] * (len(headers) - len(cols)))
                            elif len(cols) > len(headers):
                                cols = cols[:len(headers)]
                            row_dict = {}
                            for i, h in enumerate(headers):
                                row_dict[h] = cols[i]
                            table_data.append(row_dict)
                    else:
                        report_lines.extend(table_rows_buffer)
                except:
                    report_lines.extend(table_rows_buffer)
             else:
                report_lines.extend(table_rows_buffer)

        # Create Excel Writer using xlsxwriter engine for formatting
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            cell_format = workbook.add_format({
                'text_wrap': True,
                'valign': 'top',
                'border': 1
            })
            text_format = workbook.add_format({
                'text_wrap': True,
                'valign': 'top'
            })

            # --- Sheet 1: Comparison Table ---
            if table_data:
                df_table = pd.DataFrame(table_data)
                df_table.to_excel(writer, sheet_name='Comparison Table', index=False)
                worksheet = writer.sheets['Comparison Table']
                
                # Apply formatting
                for idx, col in enumerate(df_table.columns):
                    worksheet.set_column(idx, idx, 30, cell_format) # Set width to 30
                    worksheet.write(0, idx, col, header_format) # Re-write header with format
            else:
                # Create empty sheet if no table found
                pd.DataFrame({"Message": ["No structured table found in analysis."]}).to_excel(writer, sheet_name='Comparison Table', index=False)

            # --- Sheet 2: Analysis Report ---
            full_report_text = "\n".join(report_lines)
            df_report = pd.DataFrame({"Detailed Analysis Report": [full_report_text]})
            df_report.to_excel(writer, sheet_name='Detailed Analysis Report', index=False)
            
            worksheet_report = writer.sheets['Detailed Analysis Report']
            worksheet_report.set_column(0, 0, 100, text_format) # Wide column for text
            worksheet_report.write(0, 0, "Detailed Analysis Report", header_format)

        return file_path
    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        return ""

def load_document_content(file_path: str) -> str:
    """
    Helper function to load content from a file.
    """
    try:
        content = ""
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            content = "\n".join([p.page_content for p in pages])
        elif file_path.lower().endswith('.txt') or file_path.lower().endswith('.md'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif file_path.lower().endswith('.docx'):
            loader = Docx2txtLoader(file_path)
            pages = loader.load()
            content = "\n".join([p.page_content for p in pages])
        elif file_path.lower().endswith('.xlsx') or file_path.lower().endswith('.xls'):
            df = pd.read_excel(file_path)
            content = df.to_string()
        elif file_path.lower().endswith('.csv'):
             loader = CSVLoader(file_path)
             pages = loader.load()
             content = "\n".join([p.page_content for p in pages])
        return content
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return ""

def generate_summary(file_path: str) -> Dict[str, Any]:
    """
    Generate a summary for a single document.
    Extract core arguments, data, and conclusions.
    """
    try:
        content = load_document_content(file_path)
        
        if not content:
            return {"status": "error", "message": "Could not load document content."}

        # Truncate if too long (simple approach)
        if len(content) > 10000:
            content = content[:10000] + "\n...(truncated)..."

        messages = [
            {"role": "system", "content": "You are an expert document analyst. Please output in English."},
            {"role": "user", "content": f"Please provide a structured summary of the following document. \n\nRequirement:\n1. Extract core arguments.\n2. Extract key data/statistics.\n3. Extract main conclusions.\n\nDocument Content:\n{content}"}
        ]
        
        summary = call_deepseek(messages, temperature=0.5)
        return {"status": "success", "summary": summary}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def analyze_files_stream(file_paths: List[str], analysis_type: str):
    """
    Stream analysis results (summary or cross-analysis).
    """
    if analysis_type == 'summary':
        for fp in file_paths:
            if not os.path.exists(fp):
                 yield json.dumps({"type": "error", "content": f"File not found: {os.path.basename(fp)}"}) + "\n"
                 continue
            
            content = load_document_content(fp)
            if not content:
                 yield json.dumps({"type": "error", "content": f"Could not load {os.path.basename(fp)}"}) + "\n"
                 continue
                 
            if len(content) > 10000:
                content = content[:10000] + "\n...(truncated)..."
                
            messages = [
                {"role": "system", "content": "You are an expert document analyst. Please output in English."},
                {"role": "user", "content": f"Please provide a structured summary of the following document. \n\nRequirement:\n1. Extract core arguments.\n2. Extract key data/statistics.\n3. Extract main conclusions.\n\nDocument Content:\n{content}"}
            ]
            
            yield json.dumps({"type": "summary_start", "filename": os.path.basename(fp)}) + "\n"
            for token in call_deepseek_stream(messages, temperature=0.5):
                yield json.dumps({"type": "token", "content": token}) + "\n"
            yield json.dumps({"type": "summary_end", "filename": os.path.basename(fp)}) + "\n"
            
    elif analysis_type == 'cross_analysis':
        combined_content = ""
        for idx, fp in enumerate(file_paths):
            fname = os.path.basename(fp)
            content = load_document_content(fp)
            if content:
                if len(content) > 5000: content = content[:5000] + "\n...(truncated)"
                combined_content += f"\n\n--- Document {idx+1}: {fname} ---\n{content}"
        
        if not combined_content:
             yield json.dumps({"type": "error", "content": "No content loaded."}) + "\n"
             return

        prompt = f"""Please perform a deep cross-document analysis on the provided documents.
        
Tasks:
1. Identify relationships between documents (e.g., does one cite or depend on another?).
2. Contrast viewpoints: highlight differences in perspective or data.
3. Extract common conclusions shared across documents.
4. Provide a unified summary aggregating key information from all documents.
5. Create a Comparison Table (in Markdown format) comparing key metrics/views across the documents.

Please respond in English.

Documents:{combined_content}
"""
        messages = [
            {"role": "system", "content": "You are an expert researcher and analyst. Please output in English."},
            {"role": "user", "content": prompt}
        ]
        
        for token in call_deepseek_stream(messages, temperature=0.5):
            yield json.dumps({"type": "token", "content": token}) + "\n"

def delete_document(filename: str):
    """
    Delete a specific document from the vector database by filename.
    """
    try:
        if os.path.exists(CHROMA_DIR):
            vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
            # Find IDs for the document
            results = vectordb.get(where={"source": filename})
            if results and results['ids']:
                vectordb.delete(ids=results['ids'])
                return {"status": "success", "message": f"Document {filename} deleted from DB."}
            else:
                 return {"status": "warning", "message": f"Document {filename} not found in DB."}
        return {"status": "warning", "message": "Database does not exist."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to delete document: {str(e)}"}

def clear_database():
    """
    Clear the vector database.
    """
    try:
        if os.path.exists(CHROMA_DIR):
            # Try to delete the directory
            import shutil
            shutil.rmtree(CHROMA_DIR)
        return {"status": "success", "message": "Vector database cleared."}
    except Exception as e:
        # Fallback: Try to delete collection via client if directory deletion fails (e.g. due to file locks)
        try:
            vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
            vectordb.delete_collection()
            return {"status": "success", "message": "Vector database collection deleted."}
        except Exception as e2:
            return {"status": "error", "message": f"Failed to clear database: {str(e)} | {str(e2)}"}

def generate_cross_analysis(file_paths: List[str]) -> Dict[str, Any]:
    """
    Analyze multiple documents: relationships, contrasts, commonalities.
    """
    try:
        combined_content = ""
        for idx, fp in enumerate(file_paths):
            fname = os.path.basename(fp)
            content = ""
            if fp.lower().endswith('.pdf'):
                loader = PyPDFLoader(fp)
                pages = loader.load()
                content = "\n".join([p.page_content for p in pages])
            elif fp.lower().endswith('.txt') or fp.lower().endswith('.md'):
                with open(fp, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif fp.lower().endswith('.docx'):
                loader = Docx2txtLoader(fp)
                pages = loader.load()
                content = "\n".join([p.page_content for p in pages])
            elif fp.lower().endswith('.xlsx') or fp.lower().endswith('.xls'):
                df = pd.read_excel(fp)
                content = df.to_string()
            elif fp.lower().endswith('.csv'):
                loader = CSVLoader(fp)
                pages = loader.load()
                content = "\n".join([p.page_content for p in pages])
            
            # Limit per doc to avoid huge context
            if len(content) > 5000:
                content = content[:5000] + "\n...(truncated)"
            
            combined_content += f"\n\n--- Document {idx+1}: {fname} ---\n{content}"

        if not combined_content:
             return {"status": "error", "message": "No content loaded from documents."}

        prompt = f"""Please perform a deep cross-document analysis on the provided documents.
        
Tasks:
1. Identify relationships between documents (e.g., does one cite or depend on another?).
2. Contrast viewpoints: highlight differences in perspective or data.
3. Extract common conclusions shared across documents.
4. Provide a unified summary aggregating key information from all documents.
5. Create a Comparison Table (in Markdown format) comparing key metrics/views across the documents.

Please respond in English.

Documents:{combined_content}
"""
        messages = [
            {"role": "system", "content": "You are an expert researcher and analyst. Please output in English."},
            {"role": "user", "content": prompt}
        ]
        
        analysis = call_deepseek(messages, temperature=0.5)
        return {"status": "success", "analysis": analysis}
    except Exception as e:
        return {"status": "error", "message": str(e)}
