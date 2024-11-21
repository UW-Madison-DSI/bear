from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openalex_search import search

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Instruction": "Try /search?query=your_query_here&top_k=3"}


class SearchResults(BaseModel):
    name: str = Field(validation_alias="display_name")
    open_alex_url: str = Field(validation_alias="id")
    orcid: str | None
    score: float


@app.get("/search", response_model=list[SearchResults])
def search_route(
    query: str = Query(..., title="The query string to search for."),
    top_k: int = Query(3, title="The number of results to return."),
):
    if results := search(query, top_k):
        return results
    raise HTTPException(status_code=404, detail="No results found.")
