# 🤖 Agentic Pantry Robot (Vision-Language Model)

An autonomous, agentic Vision-Language Model (VLM) fine-tuned to monitor a refrigerator and trigger external Python APIs based on visual state changes.

Unlike standard VLMs that simply describe images, this model is trained for **Visual Function Calling**. It analyzes an image alongside a natural language observation and outputs a strict JSON payload to trigger inventory management tools.

## 🚀 How It Works
1. **Vision + Text Input:** The model sees a picture of the fridge and receives an observation (e.g., *"I ate the last banana"*).
2. **Reasoning:** It deduces that the inventory for bananas is now empty.
3. **Execution:** It outputs a strict JSON tool call: `{"name": "add_to_grocery_list", "arguments": {"item": "bananas", "urgency": "high"}}`

## 🛠️ Architecture & Training
* **Base Model:** `HuggingFaceTB/SmolVLM2-500M-Instruct`
* **Fine-Tuning:** Supervised Fine-Tuning (SFT) with LoRA adapters (Rank 32).
* **Hardware:** Trained on a single NVIDIA T4 GPU via Google Colab.
* **Format Forcing:** Utilized "Prefill Forcing" to guarantee deterministic JSON output, suppressing conversational hallucinations.

## 📂 Repository Structure
* `pantry_agent.py`: The main inference script containing the Python API backends and the model execution logic.
* `Agentic_VLM_Training.ipynb`: The Google Colab notebook containing the complete LoRA training pipeline, custom data collator, and bfloat16 hardware patches.
* `/PantryRobot-VLM/`: The trained LoRA adapter weights. 

## 💻 Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the agent on a local image
python pantry_agent.py
