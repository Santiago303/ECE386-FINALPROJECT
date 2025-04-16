"""This script evaluates an LLM prompt for processing text so that it can be used for the wttr.in API"""

# First, ensure the ollama package is installed
try:
    from ollama import Client
except ImportError:
    print("Error: The 'ollama' package is not installed.")
    print("Please install it using: pip install ollama")
    exit(1)

# Set the model to gemma3:27b
LLM_MODEL: str = "gemma3:27b"    # Using the 27B parameter version of Gemma

# Initialize the Ollama client
client: Client = Client(
  host='http://10.1.69.214:11434' # Default Ollama host - update if needed
)

# Verify connection and model availability
try:
    # Simple test to check connection and model availability
    response = client.generate(
        model=LLM_MODEL,
        prompt="Test connection",
        options={"temperature": 0.1}
    )
    print(f"✅ Successfully connected to Ollama server and accessed {LLM_MODEL}")
except Exception as e:
    print(f"❌ Error connecting to Ollama or accessing {LLM_MODEL}: {e}")
    print("Please check that:")
    print("1. The Ollama server is running")
    print("2. The model 'gemma3:27b' is available on your server")
    print(f"   You may need to run: ollama pull {LLM_MODEL}")
    print("3. The server URL is correct")
    exit(1)

def format_for_wttr(location: str) -> str:
    """
    Format a location string for wttr.in by replacing spaces with plus signs.
    
    Args:
        location: The extracted location string
        
    Returns:
        Properly formatted location for wttr.in URL
    """
    # Replace spaces with plus signs
    return location.replace(" ", "+")

def llm_parse_for_wttr(input_text: str) -> str:
    """
    Parse natural language weather queries into wttr.in compatible format.
    
    Args:
        input_text: Natural language query about weather
        
    Returns:
        String formatted for wttr.in API
    """
    # Clear, specific prompt with examples for better results
    prompt = f"""
    Convert this weather query to a wttr.in compatible format.
    
    FORMAT RULES:
    - City: just the city name (Example: "Paris" or "New York,US")
    - Airport: prefix with ~ (Example: "~KJFK")
    - Domain: prefix with @ (Example: "@google.com")
    - IP address: just the IP (Example: "8.8.8.8")
    - Moon phase: "/moon" or with date "/moon@DD-MM-YYYY"
    
    EXAMPLES:
    Query: "What's the weather in Tokyo?"
    Format: Tokyo
    
    Query: "Show me the weather at JFK airport"
    Format: ~KJFK
    
    Query: "What's the moon phase on January 1, 2024?"
    Format: /moon@01-01-2024
    
    YOUR TASK:
    Query: {input_text}
    Format: 
    """
    
    # Call Ollama with our prompt
    response = client.generate(
        model=LLM_MODEL,
        prompt=prompt,
        options={
            "temperature": 0.1,  # Low temperature for consistent results
            "num_predict": 50    # Limit token generation
        }
    )
    
    # Extract the response and clean it
    result = response['response'].strip()
    
    # Additional cleaning to handle potential extra text
    # If the result contains multiple lines, take only the first line
    if "\n" in result:
        result = result.split("\n")[0]
    
    # Remove any common prefixes that might appear
    prefixes_to_remove = ["Format:", "Result:", "Output:", "wttr.in format:"]
    for prefix in prefixes_to_remove:
        if result.startswith(prefix):
            result = result[len(prefix):].strip()
   
    # Format the result for wttr.in by replacing spaces with plus signs
    # But only do this for location names, not for special formats like /moon
    if not result.startswith("/"):
        result = format_for_wttr(result)
    
    return result

# Test cases for wttr.in formatting
test_cases = [
    {
        "input": "What's the weather like in New York?",
        "expected": "New+York"
    },
    {
        "input": "What's the weather like at USAFA?",
        "expected": "~USAFA"
    },
    {
        "input": "What's the weather in Colorado Springs?",
        "expected": "Colorado+Springs"
    },
    {
        "input": "Tell me the weather forecast for London, UK",
        "expected": "London,UK"
    },
    {
        "input": "Tell me the weather at Vostok Station",
        "expected": "Vostok+Station"
    },
    {
        "input": "What is the weather at the Effiel Tower",
        "expected": "Paris"
    },
    {   "input":"What is the weather in Kilimanjaro",
        "expected": "Kilimanjaro"
    },
    {
        "input": "Check weather at JFK airport",
        "expected": "~KJFK"
    },
    {   "input":"Check weather at Fort Lauderdale Airport",
        "expected": "~KFLL"
    }, 
    {
        "input": "What's the weather like at Google's headquarters?",
        "expected": "@google.com"
    },
    {
        "input": "Show me the current moon phase",
        "expected": "/moon"
    },
    {
        "input": "What will the moon look like on January 15, 2024?",
        "expected": "/moon@15-01-2024"
    },
    {
        "input": "Weather for IP 8.8.8.8",
        "expected": "8.8.8.8"
    },
]

# Function to iterate through test cases
def run_tests():
    num_passed = 0

    for i, test in enumerate(test_cases, 1):
        raw_input = test["input"]
        expected_output = test["expected"]

        print(f"\nTest {i}: {raw_input}")
        try:
            result = llm_parse_for_wttr(raw_input).strip()
            expected = expected_output.strip()

            print("LLM Output  :", result)
            print("Expected    :", expected)

            if result == expected:
                print("✅ PASS")
                num_passed += 1
            else:
                print("❌ FAIL")

        except Exception as e:
            print("💥 ERROR:", e)

    print(f"\nSummary: {num_passed} / {len(test_cases)} tests passed.")

# Run the test cases
if __name__ == "__main__":
    run_tests()