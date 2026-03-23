import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

def add_to_grocery_list(item, urgency):
    print(f"\n✅ [API SUCCESS] Added '{item}' to the Todoist Grocery List with {urgency.upper()} priority!")
    return True

def log_inventory(item, quantity):
    print(f"\n📊 [API SUCCESS] Logged {quantity} units of '{item}' into the fridge database.")
    return True

available_functions = {
    "add_to_grocery_list": add_to_grocery_list,
    "log_inventory": log_inventory
}

print("Loading Pantry Robot VLM...")
# Point this to the folder containing your downloaded LoRA weights!
MODEL_PATH = "./PantryRobot-VLM" 

processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

def analyze_fridge(image_path, user_action):
    image = Image.open(image_path).convert("RGB")
    
    tools_definition = """
    TOOLS AVAILABLE:
    1. add_to_grocery_list(item: str, urgency: str)
    2. log_inventory(item: str, quantity: float)
    
    EXAMPLE PREVIOUS INTERACTION:
    Observation: User removed last egg from carton.
    Tool Call: {"name": "add_to_grocery_list", "arguments": {"item": "eggs (12 pack)", "urgency": "high"}}
    """
    
    user_prompt = f"{tools_definition}\nObservation: {user_action}\nCRITICAL INSTRUCTION: You must output ONLY valid JSON. Do not write any conversational text.\nTool Call:"
    
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]}]
    
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    text_prompt += '{"name": "' # Force JSON generation
    
    inputs = processor(text=text_prompt, images=[image], return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
    generated_text = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()
    
    full_json_output = '{"name": "' + generated_text
    if "}" in full_json_output:
        full_json_output = full_json_output[:full_json_output.rfind("}")+1]
        
    try:
        tool_call = json.loads(full_json_output)
        func_name = tool_call.get("name")
        args = tool_call.get("arguments", {})
        
        if func_name in available_functions:
            available_functions[func_name](**args)
        else:
            print(f"❌ Unknown tool: {func_name}")
    except json.JSONDecodeError:
        print(f"❌ Failed to parse JSON: {full_json_output}")

# Example Usage (You can change this when testing locally)
if __name__ == "__main__":
    # Replace 'test.jpg' with an actual image on your computer
    # analyze_fridge("test.jpg", "I took the last banana")
    print("Script ready! Uncomment the function call above to test with an image.")
