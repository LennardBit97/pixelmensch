import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from pathlib import Path
import base64


def main():
    load_dotenv()
    client = AzureOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version="2025-04-01-preview",
        azure_endpoint="https://cidd-aifoundry-pl.openai.azure.com",
    )
    result = client.images.edit(
        image=("target.png", Path("target.png").read_bytes()),
        prompt="Transform the person in this image into a pixel figurine.",
        n=1,
        size="1536x1024",
        quality="auto",
        output_format="png",
        model="gpt-image-1",
    )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    with open("pixels.png", "wb") as f:
        f.write(image_bytes)


if __name__ == "__main__":
    load_dotenv()
    main()
