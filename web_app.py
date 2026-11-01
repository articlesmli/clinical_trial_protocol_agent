import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse

app = FastAPI(title="Clinical Trial Protocol Agent UI")

UPLOAD_DIR = "protocols"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", include_in_schema=False)
def read_root():
    return RedirectResponse(url="/docs")

@app.post("/upload-and-analyze/")
async def upload_protocol(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # 1. Save the uploaded file locally
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # 2. Trigger the agent / RAG workflow
    try:
        from main import app as agent_app
        
        initial_state = {"pdf_path": file_path}
        final_state = agent_app.invoke(initial_state)
        
        # 3. Return rich structured results to the browser
        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "message": "Protocol successfully analyzed by the RAG agent!",
            "analysis_results": final_state
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Pipeline execution failed (Check API key & file structure): {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)