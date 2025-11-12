def strip_oa_prefix(x: str) -> str:
    """Remove the OpenAlex ID prefix."""
    return x.removeprefix("https://openalex.org/").lower()
