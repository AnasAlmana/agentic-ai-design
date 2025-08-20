import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import base64

load_dotenv()

# Load OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def test_image_generation():
    """Test GPT-4.1 image generation with image_generation tool"""
    print("Testing GPT-4.1 image generation with image_generation tool...")
    
    try:
        response = client.responses.create(
            model="gpt-4.1",
            input=[{
                "role": "user", 
                "content": [
                    {"type": "input_text", "text": "Generate an image of a red apple on a white background"}
                ]
            }],
            tools=[{"type": "image_generation"}],  # Add the image generation tool
        )
        
        print("Response object type:", type(response))
        print("Response status:", response.status if hasattr(response, 'status') else 'No status')
        print("Response output type:", type(response.output))
        print("Number of output items:", len(response.output))
        
        # Save the full response for inspection
        with open("response_debug_with_tool.json", "w") as f:
            json.dump(response.model_dump(), f, indent=2, default=str)
        print("Full response saved to response_debug_with_tool.json")
        
        # Look for image_generation_call outputs
        image_generation_calls = [
            output
            for output in response.output
            if hasattr(output, 'type') and output.type == "image_generation_call"
        ]
        
        print(f"Found {len(image_generation_calls)} image generation calls")
        
        if image_generation_calls:
            for i, img_call in enumerate(image_generation_calls):
                print(f"\nImage generation call {i}:")
                print(f"  Type: {type(img_call)}")
                print(f"  Has result attr: {hasattr(img_call, 'result')}")
                if hasattr(img_call, 'result'):
                    result = img_call.result
                    print(f"  Result type: {type(result)}")
                    if isinstance(result, str):
                        print(f"  Result length: {len(result)}")
                        # Try to save the image
                        try:
                            image_bytes = base64.b64decode(result)
                            with open(f"test_apple_{i}.png", "wb") as f:
                                f.write(image_bytes)
                            print(f"  Saved image as test_apple_{i}.png")
                        except Exception as e:
                            print(f"  Error decoding image: {e}")
                    else:
                        print(f"  Result: {result}")
                
                # List all attributes
                attrs = [attr for attr in dir(img_call) if not attr.startswith('_')]
                print(f"  All attributes: {attrs}")
        else:
            print("No image generation calls found")
            # Examine all outputs
            for i, output_item in enumerate(response.output):
                print(f"\nOutput item {i}:")
                print(f"  Type: {type(output_item)}")
                print(f"  Has type attr: {hasattr(output_item, 'type')}")
                if hasattr(output_item, 'type'):
                    print(f"  Type value: {output_item.type}")
                
                if hasattr(output_item, 'content'):
                    print(f"  Content type: {type(output_item.content)}")
                    print(f"  Content length: {len(output_item.content) if output_item.content else 0}")
                    
                    # Examine content items
                    for j, content_item in enumerate(output_item.content or []):
                        print(f"    Content item {j}:")
                        print(f"      Type: {type(content_item)}")
                        if hasattr(content_item, 'type'):
                            print(f"      Type value: {content_item.type}")
                        if hasattr(content_item, 'text'):
                            print(f"      Text: {content_item.text}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_image_generation() 