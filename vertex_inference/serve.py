"""
Custom Prediction Server cho Vertex AI Endpoint.
Serve fine-tuned Llama 3.1 8B với 4-bit quantization.
Compatible với OpenAI Chat Completion API format.
"""
import os
import time
import json
import logging
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

import threading

# Global model/tokenizer
model = None
tokenizer = None
model_id = os.getenv("MODEL_ID", "thanhhoangnvbg/empathAI-llama3.1-8b")
model_lock = threading.Lock()


def load_model():
    """Load model với 4-bit quantization để fit L4 24GB."""
    global model, tokenizer
    
    if model is not None:
        return
        
    with model_lock:
        if model is not None:
            return
            
        logger.info(f"Loading model: {model_id}")
        
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir="/tmp/hf_cache",
            token=os.getenv("HF_TOKEN")
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir="/tmp/hf_cache",
            token=os.getenv("HF_TOKEN")
        )
        model.eval()
        
        logger.info("Model loaded successfully!")

# Start background model loading immediately
def background_load():
    try:
        load_model()
    except Exception as e:
        logger.error(f"Background load failed: {e}")

threading.Thread(target=background_load, daemon=True).start()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint cho Vertex AI."""
    return jsonify({"status": "healthy", "model": model_id})


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completion endpoint."""
    try:
        # Lazy load model
        if model is None:
            load_model()
        
        data = request.get_json()
        messages = data.get('messages', [])
        max_tokens = data.get('max_tokens', 512)
        temperature = data.get('temperature', 0.7)
        stream = data.get('stream', False)
        
        # Build prompt from messages
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        input_length = inputs.input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only new tokens
        new_tokens = outputs[0][input_length:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # OpenAI-compatible response format
        response = {
            "id": "chatcmpl-vertex",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text.strip()
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": input_length,
                "completion_tokens": len(new_tokens),
                "total_tokens": input_length + len(new_tokens)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def simple_chat():
    """Simple chat endpoint (compatible với Kaggle API format cũ)."""
    try:
        if model is None:
            load_model()
        
        data = request.get_json()
        messages = data.get('messages', [])
        max_tokens = data.get('max_new_tokens', 512)
        temperature = data.get('temperature', 0.7)
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        return jsonify({"reply": response_text.strip()})
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Warmup on startup
@app.before_request
def warmup():
    if model is None and request.endpoint != 'health':
        load_model()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
