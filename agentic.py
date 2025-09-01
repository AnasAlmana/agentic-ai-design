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

NUM_IMAGES = 1
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

IMPORTANT: If there is Arabic text in the image:
- Clearly identify that Arabic text is present
- Note the direction and layout of the Arabic text
- Describe the style and positioning of Arabic text elements
- Mention if there's bilingual text (Arabic with other languages)
- Preserve the exact appearance and positioning of Arabic script elements
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
creative_prompt_template = """
Your task is to generate prompt ideas for another image-image model for a given product image. 
The product is usually in the center of the image. Focus on ehnancing the product presentation and not changing the product itself.
You can change the background, scene, composition, lighting, props, angel of view, or presentation style.
Be creative and think outside the box.

Here is the image description:\n\n{description}\n\n and the actual image.

Return only JSON with keys based on the number of images: prompt1, prompt2, etc.

Start generate impressive ideas:
"""
creative_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative AI designer for marketing."),
    ("user", creative_prompt_template),
    ("user", [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{base64_image}"}}
    ])
])

def creative_with_image_preservation(input_data: dict) -> dict:
    """Wrapper to preserve base64_image through the creative step"""
    ai_message = (creative_prompt | creative_llm).invoke(input_data)
    return {
        "ai_message": ai_message,
        "base64_image": input_data["base64_image"],
        "description": input_data["description"]
    }

creative_runnable = RunnableLambda(creative_with_image_preservation)
import re

def parse_json_safe(text: str) -> dict:
    # Extract {...} from the string
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    else:
        raise ValueError("No JSON found in AI output")

# --- Step 3: Prompts -> Images using GPT-4.1 with original image ---
def generate_images_from_prompts(input_data: dict) -> dict:
    ai_message = input_data["ai_message"]
    base64_image = input_data["base64_image"]
    description = input_data.get("description", "")
    
    prompts_json = parse_json_safe(ai_message.content)
    images = {}
    
    for key, prompt in prompts_json.items():
        try:
            print(f"Generating image for {key}: {prompt}")
            enhanced_text_prompt = f"""ORIGINAL IMAGE DESCRIPTION: {description}

            ENHANCEMENT IDEA: {prompt}

            STRICT INSTRUCTIONS:
            - Generate a new image that enhances the original product presentation
            - PRESERVE the exact same product, brand name, packaging, and product identity from the original image
            - DO NOT change the product itself, its colors, shape, size, or branding
            - Only enhance: lighting, background, composition, props, angel of view, or presentation style
            - The product should remain clearly recognizable as the same item from the original image

            CRITICAL TEXT HANDLING INSTRUCTIONS:
            - Maintain all text, logos, and brand elements exactly as they appear in the original
            - For Arabic text specifically:
            * Arabic text MUST be written RIGHT-TO-LEFT (RTL direction)
            * Arabic letters MUST connect properly and maintain correct letterforms
            * Preserve Arabic script integrity with proper letter shapes (initial, medial, final, isolated forms)
            * Keep Arabic text spacing and alignment consistent with original
            * Arabic numerals should follow the correct Arabic-Indic numeral system if used in original
            * Do NOT mirror or flip Arabic text - maintain proper RTL reading direction
            * Ensure Arabic diacritics (tashkeel) are preserved if present in original
            - For any bilingual text (Arabic + English/Latin), maintain the correct direction for each script
            - Text should appear natural and readable and match exactly the original text, not distorted or backwards"""

            
            # Use GPT-4.1 with image generation tools, including original image
            response = client.responses.create(
                model="gpt-4.1",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text":enhanced_text_prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        ],
                    }
                ],
                tools=[{"type": "image_generation"}],
            )
            
            # Look for image_generation_call outputs
            image_generation_calls = [
                output
                for output in response.output
                if output.type == "image_generation_call"
            ]
            
            if image_generation_calls:
                # Get the base64 image data from the result
                image_data = image_generation_calls[0].result
                print(f"Generated image for {key}, base64 length: {len(image_data)}")
                
                # Decode and save image
                image_bytes = base64.b64decode(image_data)
                
                # Save image to file
                file_path = f"{key}_hashi.png"
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
    image_path = "images/hashi.jpg"  # path to your input image
    base64_image = encode_image(image_path)

    result = chain.invoke(base64_image)
    print(result)  # dict with {"prompt1": <image_url>, "prompt2": ..., "prompt3": ...}
