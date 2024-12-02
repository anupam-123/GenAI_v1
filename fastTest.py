from typing import Annotated
from fastapi import Body, FastAPI, File, Form, Query, Path, UploadFile
from pydantic import BaseModel
# from model_file import Main




app = FastAPI()





@app.post("/items/{prompt}")
async def read_items(result: Annotated[str | None, Query(max_length=5)] = None):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if result:
        results.update({"q": result})
    return results

@app.post('/items/{prompt_v2}')
async def read_items(item_id:Annotated[int, Path(title="The ID of the item to get")],
                     q: Annotated[str | None, Query(alias='item-query')] = None):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    return results


@app.post("/files/")
async def create_file(
    file: Annotated[bytes, File()],
    fileb: Annotated[UploadFile, File()],
    token: Annotated[str, Form()],
):
    
    return {
        "file_size": len(file),
        "token": token,
        "fileb_content_type": fileb.content_type,
    }


@app.post("/login/")
async def login(username: Annotated[str, Form()], password: Annotated[str, Form()], number: Annotated[int, Form()]):
    return {"username": username,
            "number": number}