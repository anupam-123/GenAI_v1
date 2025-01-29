# import debugpy

# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()
# print("Debugger attached")

from typing import Annotated, Optional
from fastapi import Body, FastAPI, File, Form, UploadFile
from utils import setup_logging
import shutil
from model_file import build_vector_store, query_llm, run_llm  
from fastapi.middleware.cors import CORSMiddleware
import uuid

# Dictionary to store vector stores for each session
session_store = {}

logging = setup_logging()
app = FastAPI()

# origins = [
#     "http://localhost:51699/",
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/files/")
async def create_file(
    prompt: Annotated[str, Form()],
    fileb: Optional[UploadFile] = File(None),
):
    print("promptprompt",prompt)
    logging.info("Received prompt: %s", prompt)
    if fileb is not None:
        logging.info("Received file: %s", fileb.filename)
        try:
            with open(f"uploads/{fileb.filename}", "wb") as buffer:
                shutil.copyfileobj(fileb.file, buffer)
            logging.info("File saved successfully: %s", fileb.filename)
        except Exception as e:
            logging.error(f"Failed to save file {fileb.filename}: {e}")
            return {"error": "Failed to save the uploaded file."}

    try:
        # response = {
        # "response": {
        #     "content": "Answer: Synappx Go is a mobile app that connects to Sharp multi-function printers (MFPs), shares content to Sharp displays, enables productive in-room meetings when used with Synappx Meeting, and captures workspace locations with mobile check-in.\n\nPage Numbers: 1-2.",
        #     "additional_kwargs": {},
        #     "response_metadata": {
        #     "model": "llama3.2",
        #     "created_at": "2024-12-09T05:11:48.8347837Z",
        #     "done": True,
        #     "done_reason": "stop",
        #     "total_duration": 1556532344700,
        #     "load_duration": 12400540100,
        #     "prompt_eval_count": 1420,
        #     "prompt_eval_duration": 1460827000000,
        #     "eval_count": 59,
        #     "eval_duration": 83295000000,
        #     "message": {
        #         "role": "assistant",
        #         "content": "",
        #         "images": None,
        #         "tool_calls": None
        #     }
        #     },
        #     "type": "ai",
        #     "name": None,
        #     "id": "run-33162fc1-646a-4d34-ac71-24bccf66f79b-0",
        #     "example": False,
        #     "tool_calls": [],
        #     "invalid_tool_calls": [],
        #     "usage_metadata": {
        #     "input_tokens": 1420,
        #     "output_tokens": 59,
        #     "total_tokens": 1479
        #     }
        # }
        # }
        response = run_llm(prompt)
        logging.info("LLM response: %s", response)
        return {"response": response}
    except Exception as e:
        logging.error(f"An error occurred while processing the request: {e}")
        return {"error": "An error occurred while processing your request."}