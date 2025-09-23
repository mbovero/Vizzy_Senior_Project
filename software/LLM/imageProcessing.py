import os
from openai import OpenAI
from openai import OpenAIError
from openai.types.beta.threads.message_create_params import Attachment, AttachmentToolFileSearch
import json
from dotenv import load_dotenv
import base64
from time import sleep
from timeit import default_timer as timer


# Load environment variables
load_dotenv()

imageProcessModel = "gpt-5-mini"

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

# Convert image to base64, which is a text based conversion
def convertImage(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64decode(image_file.read().decode("utf-8"))
    
def upload_image(image_path):
    print(f"Uploading image: {image_path}")
    with open (image_path, "rb") as image_file:
        results = client.files.create(
            file=image_file, purpose="vision",
        )
    print(f"Success uploading image: {image_path}")
    return results.id
    
def run_response_gpt(file_id, prompt):
    start = timer()
    print("Running Reponse GPT")
    response = client.responses.create(
        model=imageProcessModel,
        text={"verbosity":"low"},
        input=[
        {
        "role": "user",
        "content": [
            {"type": "input_text", "text": prompt},
            {
                "type": "input_image",
                "file_id": file_id,
                "detail": "low"
            },
        ],
    }],
)
    results = response.output[1].content[0].text
    #results = response.output_text
    sleep(8)
    print("Response returned")

    end = timer()
    print(f"\n Processing took {round(end-start, 2)} seconds")
    return results

prompt = """
    Your job is identify the center most image and nothing else.
    Always finish your output. Never return partial answers.
    Describe the following attributes in a couple words or 1 sentence:
    1. Name of the object
    2. Material type of the object
    3. Color of the object
    4. Any unqiue attributes of the object
    5. Where would the be most optimal position of pick up the object with a robotic claw.
    6. Give the x and y coordinates of where to pick up the object with the robotic claw relative the resolution of the image
"""



file_path1 = "white mug.png"
file_path2= "glass bottle.jpg"
file_path3="two_bottles.jpg"
file_id1 = upload_image(file_path1)
file_id2 = upload_image(file_path2)
file_id3 = upload_image(file_path3)
print(run_response_gpt(file_id1, prompt))
print(run_response_gpt(file_id2, prompt))
print(run_response_gpt(file_id3, prompt))
