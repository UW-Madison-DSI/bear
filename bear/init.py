import subprocess
from pathlib import Path

import questionary

from bear.crawler import get_openalex_id
from bear.db import init as db_init

ENV_FILE = Path(__file__).parent.parent / ".env"


def get_password(prompt: str) -> str:
    """Get password with confirmation."""
    key1, key2 = "x", "y"
    while key1 != key2:
        key1 = questionary.password(prompt).ask()
        key2 = questionary.password("Please re-enter your password/key to confirm:").ask()
        if key1 == key2:
            break
    return key1


def get_openalex_institution_id() -> str:
    """Get and confirm the institution's OpenAlex ID via user prompt."""
    while True:
        institution_name = questionary.text("What is your institution name?").ask()
        if not institution_name:
            continue

        try:
            institution_id = get_openalex_id("institutions", institution_name)
            message = f"Is this the correct ID for your institution: {institution_id}?"
            if questionary.confirm(message, default=True).ask():
                questionary.print(f"Great! We'll use this ID: {institution_id}")
                return institution_id
            else:
                questionary.print("Let's try again.")
        except ValueError:
            questionary.print(f'Institution "{institution_name}" not found. Please try again.')


def get_openalex_email() -> str:
    """Get user's email for OpenAlex polite API calls."""
    return questionary.text("What is your email address? (For OpenAlex polite API, leave empty to skip)").ask()


def use_default_milvus_settings(env_file: Path) -> None:
    """Use default Milvus settings."""
    print("Using default Milvus settings. You should change the default credentials in Production.")
    with env_file.open("a") as f:
        f.write("MINIO_ACCESS_KEY=minioadmin\n")
        f.write("MINIO_SECRET_KEY=minioadmin\n")
        f.write("MILVUS_TOKEN=root:Milvus\n")
        f.write("MILVUS_HOST=localhost\n")
        f.write("MILVUS_PORT=19530\n")
        f.write("MILVUS_DB_NAME=default\n")


def use_default_embedding_settings(env_file: Path) -> None:
    """Use default embedding settings."""
    print("Using default embedding settings.")
    with env_file.open("a") as f:
        f.write("EMBEDDING_MODEL=text-embedding-3-large\n")
        f.write("EMBEDDING_DIMS=3072\n")
        f.write("EMBEDDING_MAX_TOKENS=512\n")
        f.write("EMBEDDING_DOC_PREFIX=\n")
        f.write("EMBEDDING_QUERY_PREFIX=\n")


def quick_setup() -> None:
    confirm_quick_setup = questionary.select(
        "Quick setup for demo purpose (you can change settings in .env later)", choices=["yes", "no (you will need to manually setup .env)"]
    ).ask()

    if confirm_quick_setup == "yes":
        questionary.print("For demo purpose, we'll use as many default settings as possible to get things going.")
    else:
        questionary.print("Please manually setup the .env file.")
        return None

    if ENV_FILE.exists():
        questionary.print(
            f"There is existing environment file: {ENV_FILE}, we will not overwrite it, delete it manually if you want to recreate it with this script."
        )
        return

    # Write default settings to ENV_FILE
    use_default_milvus_settings(ENV_FILE)
    use_default_embedding_settings(ENV_FILE)

    # Get only the essentials in the quick setup
    institution_id = get_openalex_institution_id()
    email = get_openalex_email()
    api_key = get_password("Please enter your OPENAI API key (caution: We will write it to .env in plain text, make sure you are comfortable with this):")

    with ENV_FILE.open("a") as f:
        f.write(f"OPENALEX_INSTITUTION_ID={institution_id}\n")
        f.write(f"OPENALEX_MAILTO_EMAIL={email}\n")
        f.write(f"OPENAI_API_KEY={api_key}\n")

    questionary.print(f"System is configured in .env: {ENV_FILE}")


def start_backend() -> None:
    """Start the backend services."""
    subprocess.run(["docker", "compose", "up", "--build", "-d"])


def main() -> None:
    quick_setup()

    # Start backend
    if questionary.confirm("Start the backend services?", default=True).ask():
        start_backend()

    # Create DB with model.py spec
    if questionary.confirm("Initialize and wipe the database?", default=True).ask():
        db_init(wipe=True)

    # Do a test crawl
    if questionary.confirm("Start crawling?", default=True).ask():
        crawl_mode = questionary.select(
            "Select crawl mode",
            choices=["Test crawl (just download 10 people data)", "Full crawl (It may takes hours)"],
        ).ask()
        if crawl_mode == "Full crawl (It may takes hours)":
            subprocess.run(["nohup", "uv", "run", "bear/crawler.py", "&"])
        else:
            subprocess.run(["uv", "run", "bear/crawler.py", "--test"])

    # Ingest
    if questionary.confirm("Start ingesting data?", default=True).ask():
        subprocess.run(["nohup", "uv", "run", "bear/ingest.py", "&"])


if __name__ == "__main__":
    main()
