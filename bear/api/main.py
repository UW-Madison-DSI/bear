from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from bear.embedding import get_embedder
from bear.model import Work
from bear.search import SearchEngine

# Global dictionary to store shared resources
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI to manage startup and shutdown tasks."""
    app_state["search_engine"] = SearchEngine()
    yield
    app_state.clear()  # Clear the app state on shutdown


app = FastAPI(lifespan=lifespan)

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
    """Root endpoint to provide instructions for using the API."""
    return {"Instruction": "Try /search_resource?query=your_query_here&top_k=3 or /search_author?query=your_query_here&top_k=3"}


class ResourceSearchResult(BaseModel):
    id: str
    doi: str | None = None
    title: str | None = None
    display_name: str | None = None
    publication_year: int | None = None
    publication_date: str | None = None
    type: str | None = None
    cited_by_count: int | None = None
    source_display_name: str | None = None
    topics: list[str] | None = None
    abstract: str | None = None
    distance: float
    author_ids: list[str] | None = None


class AuthorSearchResult(BaseModel):
    author_id: str
    scores: dict[str, float]


class EmbedRequest(BaseModel):
    texts: list[str]
    type: Literal["query", "doc", "raw"] = "query"


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


@app.get("/search_resource", response_model=list[ResourceSearchResult])
def search_resource_route(
    query: str = Query(..., title="The query string to search for."),
    top_k: int = Query(3, title="The number of results to return."),
    resource_name: str = Query("work", title="The resource type to search (default: work)."),
    min_distance: float | None = Query(None, title="Minimum distance threshold for results."),
    since_year: int | None = Query(None, title="Filter results from this year onwards."),
):
    """Search for resources based on the provided query and parameters."""
    try:
        results = app_state["search_engine"].search_resource(
            resource_name=resource_name, query=query, top_k=top_k, min_distance=min_distance, since_year=since_year
        )

        if not results:
            raise HTTPException(status_code=404, detail="No results found.")

        # Convert results to response format
        formatted_results = []
        for result in results:
            entity = result.get("entity", {})
            # Add abstract from inverted index if available
            abstract = None
            if "abstract_inverted_index" in entity and entity["abstract_inverted_index"]:
                abstract = Work._recover_abstract(entity["abstract_inverted_index"])

            formatted_results.append(
                ResourceSearchResult(
                    id=entity.get("id", ""),
                    doi=entity.get("doi"),
                    title=entity.get("title"),
                    display_name=entity.get("display_name"),
                    publication_year=entity.get("publication_year"),
                    publication_date=entity.get("publication_date"),
                    type=entity.get("type"),
                    cited_by_count=entity.get("cited_by_count"),
                    source_display_name=entity.get("source_display_name"),
                    topics=entity.get("topics", []),
                    abstract=abstract,
                    distance=result.get("distance", 0.0),
                    author_ids=entity.get("author_ids", []),
                )
            )

        return formatted_results

    except HTTPException:
        # Re-raise HTTPExceptions (like 404) without modification
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/search_author", response_model=list[AuthorSearchResult])
def search_author_route(
    query: str = Query(..., title="The query string to search for authors."),
    top_k: int = Query(3, title="The number of results to return."),
    institutions: list[str] | None = Query(None, title="Filter authors by institutions."),
    min_distance: float | None = Query(None, title="Minimum distance threshold for results."),
    since_year: int | None = Query(None, title="Filter results from this year onwards."),
):
    """Search for authors based on the provided query and parameters."""
    try:
        results = app_state["search_engine"].search_author(
            query=query, top_k=top_k, institutions=institutions, min_distance=min_distance, since_year=since_year
        )

        if not results:
            raise HTTPException(status_code=404, detail="No results found.")

        return [AuthorSearchResult(author_id=result["author_id"], scores=result["scores"]) for result in results]

    except HTTPException:
        # Re-raise HTTPExceptions (like 404) without modification
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Author search failed: {str(e)}")


@app.post("/embed", response_model=EmbedResponse)
def embed_route(request: EmbedRequest):
    """Generate embeddings for text using the default embedding model.
    
    Args:
        request: EmbedRequest containing texts and type ("query", "doc", or "raw")
        
    Returns:
        EmbedResponse with list of embeddings
        
    The type parameter controls text preprocessing:
    - "query": Adds query prefix if configured
    - "doc": Adds document prefix if configured  
    - "raw": No prefix applied, use when you've manually added prefixes
    """
    try:
        from bear.config import config
        
        embedder = get_embedder(config.default_embedding_config)
        embeddings = embedder.embed(text=request.texts, text_type=request.type)
        
        return EmbedResponse(embeddings=embeddings)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


def main():
    uvicorn.run("bear.api.main:app", host="0.0.0.0", port=8000)
