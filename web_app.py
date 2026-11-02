# web_app.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse, RedirectResponse
from dotenv import load_dotenv
load_dotenv()

from src.workflow import app as agent_app, downstream_indexer_node

app = FastAPI(title="Clinical Trial Protocol Agent UI")

UPLOAD_DIR = "protocols"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory store for states pausing at human review
PENDING_REVIEWS = {}

@app.get("/", include_in_schema=False)
def read_root():
    return RedirectResponse(url="/docs")

@app.post("/upload-and-analyze/")
async def upload_protocol(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    try:
        initial_state = {
            "trial_id": file.filename.split(".")[0],
            "trial_title": f"Protocol {file.filename}",
            "pdf_path": file_path
        }
        final_state = agent_app.invoke(initial_state)
        
        # If paused at human review, cache it in pending reviews dictionary
        if final_state.get("review_status") == "PENDING_HUMAN_REVIEW":
            trial_id = final_state.get("trial_id", "NCT_104XB")
            PENDING_REVIEWS[trial_id] = final_state

        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "message": "Protocol analyzed and awaiting clinical sign-off!",
            "analysis_results": final_state
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Pipeline execution failed: {str(e)}"
        )

@app.get("/protocols/pending-reviews")
def get_pending_reviews():
    """Returns protocols waiting for human clinical sign-off."""
    return {"pending_reviews": PENDING_REVIEWS}

@app.post("/protocols/{trial_id}/review")
def submit_human_review(trial_id: str, approval: bool = Body(...), notes: str = Body("")):
    """Allows a clinical expert to approve or reject extracted protocol criteria."""
    if trial_id not in PENDING_REVIEWS:
        raise HTTPException(status_code=404, detail="Trial not found in pending reviews.")
    
    state = PENDING_REVIEWS[trial_id]
    
    if approval:
        state["review_status"] = "APPROVED"
        state["reviewer_notes"] = notes
        final_state = downstream_indexer_node(state)
        del PENDING_REVIEWS[trial_id]
        return {"status": "success", "message": f"Trial {trial_id} approved and indexed!", "state": final_state}
    else:
        state["review_status"] = "REJECTED"
        state["reviewer_notes"] = notes
        del PENDING_REVIEWS[trial_id]
        return {"status": "rejected", "message": f"Trial {trial_id} extraction rejected."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)