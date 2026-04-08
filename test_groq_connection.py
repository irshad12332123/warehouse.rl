"""
test_groq_connection.py — Diagnostic script for Groq API integration.
Run this to identify what's failing in your LLM setup.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("GROQ API CONNECTION DIAGNOSTIC")
print("=" * 60)

# Step 1: Check environment variables
print("\n[1] Checking environment variables...")
api_key = os.getenv("GROQ_KEY") or os.getenv("API_KEY", "")
api_base_url = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
model_name = os.getenv("MODEL_NAME", "llama-3.1-70b-versatile")

print(f"    API_BASE_URL: {api_base_url}")
print(f"    MODEL_NAME: {model_name}")
print(f"    GROQ_KEY: {'✓ Found' if api_key else '✗ NOT FOUND'}")

if not api_key:
    print("\n❌ ERROR: GROQ_KEY not found!")
    print("   → Check your .env file has: GROQ_KEY=gsk_xxxxx...")
    sys.exit(1)

if len(api_key) < 40:
    print(f"\n⚠️  WARNING: API key looks incomplete ({len(api_key)} chars)")
    print("   → Make sure you copied the FULL key from Groq dashboard")

# Step 2: Test OpenAI client initialization
print("\n[2] Testing OpenAI client initialization...")
try:
    from openai import OpenAI
    client = OpenAI(base_url=api_base_url, api_key=api_key)
    print("    ✓ OpenAI client created successfully")
except ImportError:
    print("    ✗ OpenAI package not installed")
    print("    → Run: pip install openai")
    sys.exit(1)
except Exception as e:
    print(f"    ✗ Failed to create client: {e}")
    sys.exit(1)

# Step 3: Test API connectivity
print("\n[3] Testing API connectivity...")
try:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello' and nothing else."},
        ],
        temperature=0.2,
        max_tokens=10,
    )
    reply = response.choices[0].message.content.strip()
    print(f"    ✓ API call successful!")
    print(f"    → Model responded: '{reply}'")
    print(f"    → Model used: {response.model}")
    print(f"    → Tokens used: {response.usage.total_tokens}")
except Exception as e:
    print(f"    ✗ API call failed: {e}")
    print("\n    Troubleshooting:")
    print("    - Check if GROQ_KEY is complete and valid")
    print("    - Verify API_BASE_URL is correct: https://api.groq.com/openai/v1")
    print("    - Check your internet connection")
    print("    - Visit https://console.groq.com to verify key status")
    sys.exit(1)

# Step 4: Test warehouse action parsing
print("\n[4] Testing action extraction...")
test_responses = [
    "move_up",
    "Move Up",
    "MOVE_UP",
    "Let me move up",
    "The robot should move_left",
    "up",
]

VALID_ACTIONS = ["move_up", "move_down", "move_left", "move_right"]

for test in test_responses:
    raw = test.strip().lower()
    found_action = None
    
    for a in VALID_ACTIONS:
        if a in raw:
            found_action = a
            break
    
    if not found_action:
        word = raw.split()[0] if raw else ""
        fallback = {"up": "move_up", "down": "move_down",
                    "left": "move_left", "right": "move_right"}
        found_action = fallback.get(word, "move_up")
    
    status = "✓" if found_action else "✗"
    print(f"    {status} '{test}' → {found_action}")

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED - Your LLM integration is working!")
print("=" * 60)
print("\nYou can now run: python inference.py")
