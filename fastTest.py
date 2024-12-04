from typing import Annotated
from fastapi import Body, FastAPI, File, Form, Query, Path, UploadFile
from pydantic import BaseModel
from utils import setup_logging
import shutil
from model_file import run_llm  


logging = setup_logging()
app = FastAPI()

# @app.post("/items/{prompt}")
# async def read_items(result: Annotated[str | None, Query(max_length=5)] = None):
#     results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
#     if result:
#         results.update({"q": result})
#     return results

# @app.post('/items/{prompt_v2}')
# async def read_items(item_id:Annotated[int, Path(title="The ID of the item to get")],
#                      q: Annotated[str | None, Query(alias='item-query')] = None):
#     results = {"item_id": item_id}
#     if q:
#         results.update({"q": q})
#     return results


@app.post("/files/")
async def create_file(
    fileb: Annotated[UploadFile, File()],
    prompt: Annotated[str, Form()],
):
    logging.info("File name: %s",prompt)
    logging.info("%s file loaded successfully...!",fileb.filename)
    with open(f"uploads/{fileb.filename}", "wb") as buffer:
        shutil.copyfileobj(fileb.file, buffer)
    print(prompt)
    try:
        response = run_llm(prompt)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    logging.info(response)
    return {
        response
    }
