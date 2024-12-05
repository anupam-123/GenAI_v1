from typing import Annotated
from fastapi import Body, FastAPI, File, Form, UploadFile
from utils import setup_logging
import shutil
from model_file import run_llm  
from fastapi.middleware.cors import CORSMiddleware



logging = setup_logging()
app = FastAPI()

origins = [
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/files/")
async def create_file(
    fileb: Annotated[UploadFile, File()],
    prompt: Annotated[str, Form()],
):
    logging.info("Received prompt: %s", prompt)
    logging.info("Received file: %s", fileb.filename)
    try:
        with open(f"uploads/{fileb.filename}", "wb") as buffer:
            shutil.copyfileobj(fileb.file, buffer)
        logging.info("File saved successfully: %s", fileb.filename)
    except Exception as e:
        logging.error(f"Failed to save file {fileb.filename}: {e}")
        return {"error": "Failed to save the uploaded file."}

    try:
        response = run_llm(prompt)
        logging.info("LLM response: %s", response)
        return {"response": response}
    except Exception as e:
        logging.error(f"An error occurred while processing the request: {e}")
        return {"error": "An error occurred while processing your request."}
