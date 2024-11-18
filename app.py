from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# Initialize the NVIDIA NeMo client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="YOUR_API_KEY"  # Replace with your NVIDIA API key
)

# Default route
@app.route("/", methods=["GET"])
def home():
    return "Welcome to the AI Chatbot API. Use the /chat endpoint to interact."

# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Get user input from request
        user_input = request.json.get("message")

        # Make API call to generate response
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-51b-instruct",
            messages=[{"role": "user", "content": user_input}],
            temperature=0.5,
            max_tokens=512
        )

        # Extract and return the generated content
        bot_response = completion.choices[0].message.content
        return jsonify({"response": bot_response})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
