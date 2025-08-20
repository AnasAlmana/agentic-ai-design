import base64
import json
import os
from openai import OpenAI
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage
from dotenv import load_dotenv

load_dotenv()

# Load OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Helper: encode image to base64 ---
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# --- OpenAI client ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Step 1: Image -> Description ---
IMAGE_DESCRIPTION_PROMPT_TEMPLATE = """ 
Describe the image in detail, focus on the main subject in the image usually in the center, 
extract all brand info like brand name and slogan if applicable. Make sure to include all details of the image.
"""

def describe_image(base64_image: str) -> dict:
    response = client.responses.create(
        model="gpt-4.1",  # vision-capable
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": IMAGE_DESCRIPTION_PROMPT_TEMPLATE},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                ],
            }
        ],
    )
    return {
        "description": response.output_text,
        "base64_image": base64_image
    }

image_description_runnable = RunnableLambda(describe_image)

# --- Step 2: Description + Image -> Creative Prompts ---
creative_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

creative_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative AI for marketing."),
    ("user", 
     "Here is the image description:\n\n{description}\n\n"
     "Based on this description AND the actual image, generate 3 imaginative "
     "**image generation prompts** to create new marketing visuals. "
     "Return only JSON with keys: prompt1, prompt2, prompt3. "
     "Do NOT include any extra text or explanations."),
    ("user", [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{base64_image}"}}
    ])
])


creative_runnable = creative_prompt | creative_llm
import re

def parse_json_safe(text: str) -> dict:
    # Extract {...} from the string
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    else:
        raise ValueError("No JSON found in AI output")

# --- Step 3: Prompts -> Images using GPT-4.1 with image_generation tool ---
def generate_images_from_prompts(ai_message: AIMessage) -> dict:
    prompts_json = parse_json_safe(ai_message.content)
    images = {}
    
    for key, prompt in prompts_json.items():
        try:
            print(f"Generating image for {key}: {prompt}")
            
            # Use GPT-4.1 with image_generation tool
            response = client.responses.create(
                model="gpt-4.1",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": f"Generate an image: {prompt}"}
                        ]
                    }
                ],
                tools=[{"type": "image_generation"}],  # Enable image generation tool
            )
            
            # Look for image_generation_call outputs
            image_generation_calls = [
                output
                for output in response.output
                if hasattr(output, 'type') and output.type == "image_generation_call"
            ]
            
            if image_generation_calls:
                # Get the base64 image data from the result
                image_data = image_generation_calls[0].result
                print(f"Generated image for {key}, base64 length: {len(image_data)}")
                
                # Decode and save image
                image_bytes = base64.b64decode(image_data)
                
                # Save image to file
                file_path = f"{key}.png"
                with open(file_path, "wb") as f:
                    f.write(image_bytes)
                
                images[key] = file_path
                print(f"Saved image to {file_path}")
            else:
                print(f"No image generated for {key}. Response output:")
                for output in response.output:
                    print(f"  Output type: {getattr(output, 'type', 'unknown')}")
                    if hasattr(output, 'content'):
                        for content in output.content:
                            if hasattr(content, 'text'):
                                print(f"  Text: {content.text}")
                images[key] = "No image generated"
                
        except Exception as e:
            print(f"Error generating image for {key}: {e}")
            import traceback
            traceback.print_exc()
            images[key] = f"Error: {str(e)}"
            
    return images

image_generation_runnable = RunnableLambda(generate_images_from_prompts)

# --- Full Chain: Image -> Description -> Creative Prompts -> Generated Images ---
chain = image_description_runnable | creative_runnable | image_generation_runnable

# --- Run the chain ---
if __name__ == "__main__":
    image_path = "images/img.jpg"  # path to your input image
    base64_image = encode_image(image_path)

    result = chain.invoke(base64_image)
    print(result)  # dict with {"prompt1": <image_url>, "prompt2": ..., "prompt3": ...}
