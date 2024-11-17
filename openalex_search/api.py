from fastapi import FastAPI, HTTPException, Query

from openalex_search import search

app = FastAPI()


@app.get("/")
def read_root():
    return {"Instruction": "Try /search?query=your_query_here&top_k=3"}


@app.get("/search")
def search_route(
    query: str = Query(..., title="The query string to search for."),
    top_k: int = Query(3, title="The number of results to return."),
):
    if results := search(query, top_k):
        return results
    raise HTTPException(status_code=404, detail="No results found.")
