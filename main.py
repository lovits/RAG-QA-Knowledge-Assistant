from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from backend.models import UserLogin, UserRegister, User, UserUpdate
from backend.database import init_db, get_user, create_user, verify_login, update_user, get_all_users, delete_user
from backend.rag import ingest_documents, get_answer, process_uploaded_file, generate_summary, generate_cross_analysis, clear_database, export_cross_analysis_to_excel, get_answer_stream, analyze_files_stream, delete_document, extract_user_persona
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import shutil
from fastapi.responses import FileResponse, StreamingResponse
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB on startup
    init_db()
    yield

app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route to serve index.html
@app.get("/")
async def read_root():
    response = FileResponse('static/index.html')
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Mount static files if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static", html=True), name="static")
    
# Mount exports directory
if not os.path.exists("exports"):
    os.makedirs("exports")
app.mount("/exports", StaticFiles(directory="exports"), name="exports")

# Chat Models
class ChatRequest(BaseModel):
    question: str
    history: List[dict] = []
    username: Optional[str] = "admin" # Default to admin if not provided
    
class ExportRequest(BaseModel):
    content: str
    type: str = "excel"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    username = request.username or "admin"
    current_user = get_user(username)
    user_persona = current_user.get("persona", "") if current_user else ""
    
    # Check if we should update persona
    new_persona_info = extract_user_persona(request.question)
    if new_persona_info:
        print(f"Detected new persona info: {new_persona_info}")
        if user_persona:
            # Simple check to avoid exact duplicates
            if new_persona_info not in user_persona:
                updated_persona = f"{user_persona}\n{new_persona_info}"
                update_user(username, {"persona": updated_persona})
                user_persona = updated_persona
        else:
            updated_persona = new_persona_info
            update_user(username, {"persona": updated_persona})
            user_persona = updated_persona

    return StreamingResponse(get_answer_stream(request.question, request.history, user_persona), media_type="application/x-ndjson")
    content: str
    type: str = "excel"

@app.post("/api/register")
def register(user: UserRegister):
    if not user.username or not user.username.strip():
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    if not user.password or len(user.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
        
    success = create_user(
        user.username.strip(), 
        user.password, 
        user.email, 
        user.phone,
        user.avatar,
        user.preferred_output_format
    )
    if not success:
        raise HTTPException(status_code=400, detail="Username already exists")
    return {"message": "Registration successful"}

@app.post("/api/login")
def login(user: UserLogin):
    if not user.username or not user.username.strip():
        raise HTTPException(status_code=400, detail="Please enter username")
    if not user.password:
        raise HTTPException(status_code=400, detail="Please enter password")

    result = verify_login(user.username.strip(), user.password)
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["message"])
    
    # Return basic user info
    db_user = get_user(user.username)
    return {
        "message": "Login successful", 
        "username": user.username,
        "avatar": db_user.get("avatar"),
        "preferred_output_format": db_user.get("preferred_output_format", "markdown")
    }

@app.get("/api/user/{username}", response_model=User)
def get_user_profile(username: str):
    user = get_user(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # Don't return password or internal fields
    return User(
        username=user['username'],
        email=user['email'],
        phone=user['phone'],
        avatar=user['avatar'],
        preferred_output_format=user['preferred_output_format']
    )

@app.put("/api/user/{username}")
def update_user_profile(username: str, update_data: UserUpdate):
    # In a real app, we would verify the current user matches 'username' via token
    # For this demo, we assume the client is authorized
    
    # Validate password if present
    if update_data.password is not None:
        if len(update_data.password) < 6:
             raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    success = update_user(username, update_data.dict(exclude_unset=True))
    if not success:
        raise HTTPException(status_code=400, detail="Update failed")
    return {"message": "Profile updated successfully"}

@app.get("/api/admin/users", response_model=List[dict])
def list_users():
    return get_all_users()

@app.delete("/api/admin/users/{username}")
def delete_user_account(username: str):
    success = delete_user(username)
    if not success:
        raise HTTPException(status_code=404, detail="User not found or delete failed")
    return {"message": f"User {username} deleted successfully"}

@app.post("/api/ingest")
def ingest():
    result = ingest_documents()
    return result

class AnalysisRequest(BaseModel):
    filenames: List[str]
    type: str # 'summary' or 'cross_analysis'

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    results = []
    for file in files:
        file_path = os.path.join(upload_dir, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process and ingest each file
            res = process_uploaded_file(file_path)
            results.append({"filename": file.filename, "status": res["status"], "message": res["message"]})
        except Exception as e:
            results.append({"filename": file.filename, "status": "error", "message": str(e)})
            
    return {"results": results}

@app.get("/api/files")
def list_files():
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        return {"files": []}
    
    files = [f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]
    return {"files": files}

@app.delete("/api/files/{filename}")
def delete_single_file(filename: str):
    upload_dir = "uploads"
    file_path = os.path.join(upload_dir, filename)
    
    # Delete from disk
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete file from disk: {str(e)}")
    else:
        raise HTTPException(status_code=404, detail="File not found")
        
    # Delete from Vector DB
    db_res = delete_document(filename)
    
    return {
        "message": f"File {filename} deleted successfully",
        "db_status": db_res
    }

@app.delete("/api/files")
def delete_all_files():
    # 1. Clear Vector DB
    db_res = clear_database()
    
    # 2. Delete Uploaded Files
    upload_dir = "uploads"
    deleted_files_count = 0
    if os.path.exists(upload_dir):
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    deleted_files_count += 1
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    deleted_files_count += 1
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    
    return {
        "message": "All files and data cleared.",
        "db_status": db_res,
        "deleted_files": deleted_files_count
    }

@app.post("/api/analyze")
def analyze(request: AnalysisRequest):
    upload_dir = "uploads"
    file_paths = [os.path.join(upload_dir, f) for f in request.filenames]
    
    # Validate files exist
    valid_paths = [fp for fp in file_paths if os.path.exists(fp)]
    
    if not valid_paths:
        raise HTTPException(status_code=400, detail="No valid analysis files found")

    if request.type in ['summary', 'cross_analysis']:
         return StreamingResponse(analyze_files_stream(valid_paths, request.type), media_type="application/x-ndjson")
    
    else:
        raise HTTPException(status_code=400, detail="Invalid analysis type")



@app.post("/api/export")
def export_data(request: ExportRequest):
    if request.type == "excel":
        file_path = export_cross_analysis_to_excel(request.content)
        if file_path and os.path.exists(file_path):
            filename = os.path.basename(file_path)
            return {"url": f"/exports/{filename}", "filename": filename}
        else:
            raise HTTPException(status_code=500, detail="Export failed")
    else:
        raise HTTPException(status_code=400, detail="Unsupported export type")

if __name__ == "__main__":
    print("Server starting at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
