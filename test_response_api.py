#!/usr/bin/env python
"""
Test script for the message response API endpoint
"""

import requests
import json
import sys
import uuid

# Base URL for API calls
BASE_URL = "http://localhost:8888"

# ANSI Colors for pretty output
COLORS = {
    'GREEN': '\033[92m',
    'RED': '\033[91m',
    'YELLOW': '\033[93m',
    'BLUE': '\033[94m',
    'ENDC': '\033[0m',
}

def print_colored(text, color='GREEN'):
    """Print colored text"""
    print(f"{COLORS.get(color, '')}{text}{COLORS['ENDC']}")

def test_message_response_flow():
    """Test the complete message and response flow"""
    print_colored("\nTESTING MESSAGE RESPONSE FLOW", 'BLUE')
    
    # 1. Create a new message
    message_data = {
        "content": "What is the capital of France?",
        "metadata": {
            "source": "test_script",
            "test_id": str(uuid.uuid4())
        }
    }
    
    print_colored("\n1. Creating original message...", 'YELLOW')
    try:
        response = requests.post(f"{BASE_URL}/messages/", json=message_data)
        if response.status_code == 200:
            message = response.json()
            message_id = message['id']
            print_colored(f"\u2713 Message created with ID: {message_id}", 'GREEN')
        else:
            print_colored(f"\u2717 Failed to create message: {response.text}", 'RED')
            return
    except Exception as e:
        print_colored(f"\u2717 Error: {str(e)}", 'RED')
        return
    
    # 2. Create a response to the message
    response_data = {
        "content": "The capital of France is Paris.",
        "metadata": {
            "agent_id": "test_agent",
            "agent": "Test Response Agent",
            "response_type": "generated"
        }
    }
    
    print_colored("\n2. Creating response to the message...", 'YELLOW')
    try:
        response = requests.post(f"{BASE_URL}/messages/{message_id}/responses", json=response_data)
        if response.status_code == 200:
            response_message = response.json()
            response_id = response_message['id']
            print_colored(f"\u2713 Response created with ID: {response_id}", 'GREEN')
            print_colored(f"\u2713 Response content: {response_message['content']}", 'GREEN')
            print_colored(f"\u2713 Linked to original message: {response_message['metadata'].get('response_to')}", 'GREEN')
        else:
            print_colored(f"\u2717 Failed to create response: {response.text}", 'RED')
            return
    except Exception as e:
        print_colored(f"\u2717 Error: {str(e)}", 'RED')
        return
    
    # 3. Get responses for the original message
    print_colored("\n3. Getting responses for the original message...", 'YELLOW')
    try:
        response = requests.get(f"{BASE_URL}/messages/{message_id}/responses")
        if response.status_code == 200:
            responses = response.json()
            if responses and len(responses) > 0:
                print_colored(f"\u2713 Found {len(responses)} response(s) for message {message_id}", 'GREEN')
                for idx, resp in enumerate(responses):
                    print_colored(f"  {idx+1}. {resp['content']} (ID: {resp['id']})", 'GREEN')
            else:
                print_colored(f"\u2717 No responses found for message {message_id}", 'RED')
        else:
            print_colored(f"\u2717 Failed to get responses: {response.text}", 'RED')
    except Exception as e:
        print_colored(f"\u2717 Error: {str(e)}", 'RED')
    
    print_colored("\nMESSAGE RESPONSE FLOW TEST COMPLETE", 'BLUE')

if __name__ == "__main__":
    test_message_response_flow()
