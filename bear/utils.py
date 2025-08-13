def strip_oa_prefix(x: str) -> str:
    """Remove the OpenAlex ID prefix."""
    return x.lstrip("https://openalex.org/").lower()
