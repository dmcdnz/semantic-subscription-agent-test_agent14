# Semantic Subscription System API Documentation

This document provides a comprehensive reference for all API endpoints available in the Semantic Subscription System. The system exposes a RESTful API for message management, agent operations, containerized deployment, and GitHub integration.

## Table of Contents

1. [Core Message API](#1-core-message-api)
2. [Agent Management API](#2-agent-management-api)
3. [Containerized Agent Deployment API](#3-containerized-agent-deployment-api)
4. [GitHub Integration API](#4-github-integration-api)
5. [Memory API](#5-memory-api)
6. [LLM API](#6-llm-api)
7. [Authentication API](#7-authentication-api)
8. [Web Portal API](#8-web-portal-api)

## API Base URLs

- **Core API**: `http://<host>:8888/api`
- **Agent Service API**: `http://<host>:8889/api`

## Authentication

Many endpoints require authentication through GitHub OAuth. Protected endpoints will redirect unauthenticated requests to the login page.

---

## 1. Core Message API

These endpoints manage the core message operations including creating messages, retrieving messages, and performing semantic searches.

### Create a New Message

- **URL**: `/messages/`
- **Method**: `POST`
- **Description**: Creates a new message in the system, which triggers the interest determination phase

**Request Body**:
```json
{
  "content": "Message content text",
  "metadata": {
    "source": "api",
    "user_id": "user123",
    "tags": ["example", "documentation"]
  }
}
```

**Response**:
```json
{
  "id": "msg_12345",
  "content": "Message content text",
  "metadata": {
    "source": "api",
    "user_id": "user123",
    "tags": ["example", "documentation"]
  },
  "created_at": "2025-04-23T12:34:56",
  "processed_by": [],
  "response_ids": []
}
```

### Get a Message by ID

- **URL**: `/messages/{message_id}`
- **Method**: `GET`
- **Description**: Retrieves a specific message by its ID

**Response**:
```json
{
  "id": "msg_12345",
  "content": "Message content text",
  "metadata": {...},
  "created_at": "2025-04-23T12:34:56",
  "processed_by": ["agent_789", "agent_456"],
  "response_ids": ["msg_67890"]
}
```

### Get Message Responses

- **URL**: `/messages/{message_id}/responses`
- **Method**: `GET`
- **Description**: Retrieves all responses to a specific message

**Response**:
```json
[
  {
    "id": "msg_67890",
    "content": "Response from an agent",
    "metadata": {
      "agent_id": "agent_789",
      "response_type": "generated"
    },
    "created_at": "2025-04-23T12:35:20"
  }
]
```

### Search Messages

- **URL**: `/messages/search`
- **Method**: `POST`
- **Description**: Performs a semantic search to find messages similar to the query

**Request Body**:
```json
{
  "query": "find messages about machine learning",
  "limit": 10,
  "threshold": 0.7
}
```

**Response**:
```json
[
  {
    "id": "msg_12345",
    "content": "Introduction to machine learning algorithms",
    "similarity": 0.92,
    "metadata": {...},
    "created_at": "2025-04-20T15:22:31"
  },
  {
    "id": "msg_23456",
    "content": "Comparing different ML frameworks",
    "similarity": 0.87,
    "metadata": {...},
    "created_at": "2025-04-21T09:12:48"
  }
]
```

---

## 2. Agent Management API

These endpoints manage the lifecycle of agents in the system.

### List All Agents

- **URL**: `/agents/`
- **Method**: `GET`
- **Description**: Lists all registered agents in the system

**Response**:
```json
{
  "agents": [
    {
      "agent_id": "agent_123",
      "name": "Customer Support Agent",
      "description": "Handles customer inquiries",
      "status": "running",
      "class_name": "CustomerSupportAgent"
    },
    {
      "agent_id": "agent_456",
      "name": "Data Analysis Agent",
      "description": "Analyzes data messages",
      "status": "running",
      "class_name": "DataAnalysisAgent"
    }
  ]
}
```

### Get Agent Details

- **URL**: `/agents/{agent_id}`
- **Method**: `GET`
- **Description**: Gets detailed information about a specific agent

**Response**:
```json
{
  "agent_id": "agent_123",
  "name": "Customer Support Agent",
  "description": "Handles customer inquiries",
  "class_name": "CustomerSupportAgent",
  "is_running": true,
  "repo_url": "https://github.com/org/customer-support-agent",
  "interest_areas": ["customer", "support", "help", "issue"]
}
```

### Create a New Agent

- **URL**: `/api/agents`
- **Method**: `POST`
- **Description**: Creates a new agent from a class

**Request Body**:
```json
{
  "class_name": "CustomerSupportAgent",
  "name": "Premium Support Agent",
  "description": "Handles premium customer inquiries",
  "config": {
    "response_template": "Thank you for contacting premium support...",
    "priority": "high"
  }
}
```

**Response**:
```json
{
  "agent_id": "agent_789",
  "name": "Premium Support Agent",
  "description": "Handles premium customer inquiries",
  "class_name": "CustomerSupportAgent",
  "is_running": false
}
```

### Start Agent

- **URL**: `/agents/{agent_id}/start`
- **Method**: `POST`
- **Description**: Starts the agent's processing thread

**Response**:
```json
{
  "status": "success",
  "message": "Agent agent_789 started successfully"
}
```

### Stop Agent

- **URL**: `/agents/{agent_id}/stop`
- **Method**: `POST`
- **Description**: Stops the agent's processing thread

**Response**:
```json
{
  "status": "success",
  "message": "Agent agent_789 stopped successfully"
}
```

### Process Message With Agent

- **URL**: `/agents/{agent_id}/process`
- **Method**: `POST`
- **Description**: Processes a message directly with a specific agent

**Request Body**:
```json
{
  "content": "I need help with my subscription",
  "metadata": {
    "source": "direct_api_call",
    "priority": "high"
  }
}
```

**Response**:
```json
{
  "message_id": "msg_34567",
  "response": {
    "id": "msg_45678",
    "content": "I'd be happy to help with your subscription issue. What specific problem are you experiencing?",
    "metadata": {
      "agent_id": "agent_789",
      "response_type": "generated"
    },
    "created_at": "2025-04-23T14:22:10"
  }
}
```

---

## 3. Containerized Agent Deployment API

These endpoints manage containerized agent deployments.

### Deploy Agent from GitHub

- **URL**: `/api/agents/deploy`
- **Method**: `POST`
- **Description**: Deploys an agent directly from a GitHub repository
- **Authentication**: Requires GitHub authentication

**Request Body**:
```json
{
  "user_id": "github_user123",
  "repo_url": "https://github.com/user/agent-repo",
  "branch": "main",
  "agent_name": "Custom Agent",
  "configuration": {
    "model_type": "semantic",
    "interest_threshold": 0.75
  }
}
```

**Response**:
```json
{
  "deployment_id": "deploy_12345",
  "agent_id": "agent_567",
  "agent_name": "Custom Agent",
  "status": "building",
  "container_id": null,
  "image_name": "semsubscription/agent-567:latest"
}
```

### Get Deployment Status

- **URL**: `/api/agents/deployments/{deployment_id}`
- **Method**: `GET`
- **Description**: Gets the current status of an agent deployment
- **Authentication**: Requires GitHub authentication

**Response**:
```json
{
  "deployment_id": "deploy_12345",
  "agent_id": "agent_567",
  "agent_name": "Custom Agent",
  "status": "running",
  "container_id": "a1b2c3d4e5f6",
  "logs": "Agent starting...\nConnected to event bus\nRegistered agent interests\nListening for events\n"
}
```

### Stop Deployment

- **URL**: `/api/agents/deployments/{deployment_id}`
- **Method**: `DELETE`
- **Description**: Stops and removes an agent deployment
- **Authentication**: Requires GitHub authentication

**Response**:
```json
{
  "status": "success",
  "message": "Deployment deploy_12345 stopped"
}
```

### List Deployments

- **URL**: `/api/agents/deployments`
- **Method**: `GET`
- **Description**: Lists all agent deployments, optionally filtered by user ID
- **Authentication**: Requires GitHub authentication

**Query Parameters**:
- `user_id`: (Optional) Filter deployments by user ID

**Response**:
```json
[
  {
    "deployment_id": "deploy_12345",
    "agent_id": "agent_567",
    "agent_name": "Custom Agent",
    "status": "running",
    "container_id": "a1b2c3d4e5f6",
    "image_name": "semsubscription/agent-567:latest"
  },
  {
    "deployment_id": "deploy_23456",
    "agent_id": "agent_678",
    "agent_name": "Another Agent",
    "status": "failed",
    "container_id": null,
    "image_name": "semsubscription/agent-678:latest"
  }
]
```

---

## 4. GitHub Integration API

These endpoints manage GitHub repository integration for agents.

### Create Agent Repository

- **URL**: `/api/agents/create-repo`
- **Method**: `POST`
- **Description**: Creates a new agent repository on GitHub from a template
- **Authentication**: Requires GitHub authentication

**Request Body**:
```json
{
  "name": "sales-assistant-agent",
  "domain": "sales",
  "description": "Agent for helping with sales inquiries",
  "github_token": "<github_token>"
}
```

**Response**:
```json
{
  "status": "success",
  "repo_url": "https://github.com/user/sales-assistant-agent",
  "message": "Successfully created repository for sales-assistant-agent"
}
```

### Update Agent from Repository

- **URL**: `/api/agents/{agent_id}/update`
- **Method**: `POST`
- **Description**: Updates an agent from its GitHub repository
- **Authentication**: Requires GitHub authentication

**Response**:
```json
{
  "status": "success",
  "message": "Agent agent_567 updated successfully"
}
```

---

## 5. Memory API

These endpoints manage the vector memory system for storing and retrieving contextual information.

### Create a Memory

- **URL**: `/api/memories/`
- **Method**: `POST`
- **Description**: Creates a new memory in the vector memory system

**Request Body**:
```json
{
  "content": "This is important information to remember",
  "metadata": {
    "source": "documentation",
    "priority": "high"
  },
  "tags": ["documentation", "example", "contextual"]
}
```

**Response**:
```json
{
  "memory_id": "mem_12345",
  "content": "This is important information to remember",
  "metadata": {
    "source": "documentation",
    "priority": "high"
  },
  "created_at": "2025-04-23T12:34:56",
  "updated_at": "2025-04-23T12:34:56",
  "tags": ["documentation", "example", "contextual"]
}
```

### Get a Memory by ID

- **URL**: `/api/memories/{memory_id}`
- **Method**: `GET`
- **Description**: Retrieves a specific memory by its ID

**Response**:
```json
{
  "memory_id": "mem_12345",
  "content": "This is important information to remember",
  "metadata": {
    "source": "documentation",
    "priority": "high"
  },
  "created_at": "2025-04-23T12:34:56",
  "updated_at": "2025-04-23T12:34:56",
  "tags": ["documentation", "example", "contextual"]
}
```

### Update a Memory

- **URL**: `/api/memories/{memory_id}`
- **Method**: `PUT`
- **Description**: Updates an existing memory

**Request Body**:
```json
{
  "content": "Updated information to remember",
  "metadata": {
    "source": "update",
    "priority": "medium"
  },
  "tags": ["updated", "example"]
}
```

**Response**:
```json
{
  "memory_id": "mem_12345",
  "content": "Updated information to remember",
  "metadata": {
    "source": "update",
    "priority": "medium"
  },
  "created_at": "2025-04-23T12:34:56",
  "updated_at": "2025-04-24T14:12:33",
  "tags": ["updated", "example"]
}
```

### Delete a Memory

- **URL**: `/api/memories/{memory_id}`
- **Method**: `DELETE`
- **Description**: Deletes a memory from the system

**Response**:
```json
{
  "status": "success",
  "message": "Memory mem_12345 deleted"
}
```

### Search Memories

- **URL**: `/api/memories/search`
- **Method**: `POST`
- **Description**: Performs a semantic search to find memories similar to the query

**Request Body**:
```json
{
  "query": "find information about documentation",
  "limit": 5,
  "min_similarity": 0.7,
  "tags": ["documentation"]
}
```

**Response**:
```json
[
  {
    "memory_id": "mem_12345",
    "content": "This is important information to remember",
    "metadata": {
      "source": "documentation",
      "priority": "high"
    },
    "created_at": "2025-04-23T12:34:56",
    "updated_at": "2025-04-23T12:34:56",
    "tags": ["documentation", "example", "contextual"]
  },
  {
    "memory_id": "mem_23456",
    "content": "More information about system documentation",
    "metadata": {
      "source": "api",
      "priority": "medium"
    },
    "created_at": "2025-04-22T09:12:34",
    "updated_at": "2025-04-22T09:12:34",
    "tags": ["documentation", "system"]
  }
]
```

### Get Memories by Tag

- **URL**: `/api/memories/tag/{tag}`
- **Method**: `GET`
- **Description**: Retrieves memories with a specific tag

**Query Parameters**:
- `limit`: (Optional) Maximum number of memories to return (default 100)

**Response**:
```json
[
  {
    "memory_id": "mem_12345",
    "content": "This is important information to remember",
    "metadata": {
      "source": "documentation",
      "priority": "high"
    },
    "created_at": "2025-04-23T12:34:56",
    "updated_at": "2025-04-23T12:34:56",
    "tags": ["documentation", "example", "contextual"]
  }
]
```

### Get Recent Memories

- **URL**: `/api/memories/`
- **Method**: `GET`
- **Description**: Retrieves the most recent memories

**Query Parameters**:
- `limit`: (Optional) Maximum number of memories to return (default 10)

**Response**:
```json
[
  {
    "memory_id": "mem_34567",
    "content": "Latest information added to the system",
    "metadata": {
      "source": "user",
      "priority": "normal"
    },
    "created_at": "2025-04-24T10:45:12",
    "updated_at": "2025-04-24T10:45:12",
    "tags": ["recent", "user-generated"]
  },
  {
    "memory_id": "mem_12345",
    "content": "This is important information to remember",
    "metadata": {
      "source": "documentation",
      "priority": "high"
    },
    "created_at": "2025-04-23T12:34:56",
    "updated_at": "2025-04-23T12:34:56",
    "tags": ["documentation", "example", "contextual"]
  }
]
```

---

## 6. LLM API

These endpoints provide access to the core system's LLM (Large Language Model) capabilities, allowing containerized agents to use language models without implementing their own integrations.

### Generate Text

- **URL**: `/api/llm/generate`
- **Method**: `POST`
- **Description**: Generates text from a prompt using the specified LLM provider and model

**Request Body**:
```json
{
  "prompt": "Write a short summary of semantic search technology.",
  "model": "gpt-3.5-turbo-instruct",
  "provider": "openai",
  "temperature": 0.7,
  "max_tokens": 250,
  "agent_id": "agent_123",
  "params": {
    "top_p": 1.0
  }
}
```

**Response**:
```json
{
  "text": "Semantic search technology goes beyond traditional keyword matching by understanding the meaning and context of search queries. It leverages natural language processing and vector embeddings to capture semantic relationships between words and concepts. This allows systems to return results based on conceptual similarity rather than exact word matches, significantly improving search relevance. Modern semantic search implementations often use transformer models to convert text into high-dimensional vectors, enabling efficient similarity comparisons in vector space.",
  "model": "gpt-3.5-turbo-instruct",
  "provider": "openai",
  "usage": {}
}
```

### Chat Completion

- **URL**: `/api/llm/chat`
- **Method**: `POST`
- **Description**: Generates a response to a conversational context using a chat model

**Request Body**:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant that explains technical concepts clearly."
    },
    {
      "role": "user",
      "content": "How does vector similarity search work?"
    }
  ],
  "model": "gpt-3.5-turbo",
  "provider": "openai",
  "temperature": 0.5,
  "max_tokens": 300,
  "agent_id": "agent_456"
}
```

**Response**:
```json
{
  "text": "Vector similarity search works by converting data (like text, images, or audio) into numerical vectors in a high-dimensional space, where each dimension represents a feature or attribute. This process, called embedding, transforms semantic meaning into mathematical representations.

When performing a search:

1. Your query is converted into the same vector format
2. The system calculates the 'distance' or similarity between your query vector and all the stored vectors
3. Results are ranked by similarity scores, with closer vectors being more relevant

Common similarity metrics include cosine similarity (measuring the angle between vectors), Euclidean distance, or dot product. To make this efficient at scale, approximate nearest neighbor algorithms like HNSW, FAISS, or Annoy are used.

This approach enables semantic search that understands meaning beyond keywords, powers recommendation systems, and forms the foundation for many AI applications.",
  "model": "gpt-3.5-turbo",
  "provider": "openai",
  "usage": {
    "prompt_tokens": 29,
    "completion_tokens": 173,
    "total_tokens": 202
  }
}
```

### Generate Embeddings

- **URL**: `/api/llm/embed`
- **Method**: `POST`
- **Description**: Generates vector embeddings for text using the specified LLM provider and model

**Request Body**:
```json
{
  "text": "This is a sample text for embedding generation.",
  "model": "text-embedding-3-small",
  "provider": "openai",
  "agent_id": "agent_789"
}
```

**Response**:
```json
{
  "embeddings": [0.0253, -0.0204, 0.0676, ...], // truncated for brevity (usually 1536 dimensions)
  "model": "text-embedding-3-small",
  "provider": "openai",
  "dimensions": 1536
}
```

Note: You can also embed multiple texts in a single request by providing an array of strings for the `text` field:

```json
{
  "text": ["First text to embed", "Second text to embed"],
  "model": "text-embedding-3-small",
  "provider": "openai"
}
```

The response will contain a nested array of embeddings.

### List Available Models

- **URL**: `/api/llm/models`
- **Method**: `POST`
- **Description**: Lists available models from the specified provider

**Request Body**:
```json
{
  "provider": "openai",
  "agent_id": "agent_123"
}
```

**Response**:
```json
{
  "models": [
    {
      "id": "gpt-4o",
      "created": 1699541136,
      "owned_by": "openai",
      "object": "model"
    },
    {
      "id": "gpt-4-turbo",
      "created": 1699374443,
      "owned_by": "openai",
      "object": "model"
    },
    {
      "id": "gpt-3.5-turbo",
      "created": 1699351999,
      "owned_by": "openai",
      "object": "model"
    }
    // Additional models omitted for brevity
  ],
  "provider": "openai"
}
```

### Example Agent Implementation

Here's an example of how a containerized agent can use the LLM API to process messages:

```python
def process_message(self, message):
    # Extract content
    content = getattr(message, 'content', '') if not isinstance(message, dict) else message.get('content', '')
    
    try:
        import requests
        
        # Core API URL (from environment variable or config)
        core_api_url = os.environ.get("CORE_API_URL", "http://host.docker.internal:8888")
        
        # Call the LLM API endpoint for chat
        response = requests.post(
            f"{core_api_url}/api/llm/chat",
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content}
                ],
                "model": "gpt-4o",  # Use your preferred model
                "agent_id": self.agent_id
            }
        )
        
        if response.status_code == 200:
            llm_response = response.json()
            return {
                "agent": self.name,
                "response": llm_response.get("text", ""),
                "model_used": llm_response.get("model", "unknown")
            }
    except Exception as e:
        # Error handling
        return {"error": str(e)}
```

---

## 7. Authentication API

These endpoints handle GitHub authentication for the system.

### GitHub Login

- **URL**: `/api/auth/github`
- **Method**: `GET`
- **Description**: Redirects to GitHub for OAuth authentication

### GitHub Callback

- **URL**: `/api/auth/github/callback`
- **Method**: `GET`
- **Description**: Handles the callback from GitHub after authentication

### Logout

- **URL**: `/api/auth/logout`
- **Method**: `GET`
- **Description**: Logs out the user by clearing the session

### Authentication Status

- **URL**: `/api/auth/status`
- **Method**: `GET`
- **Description**: Gets the current authentication status and user info

**Response**:
```json
{
  "authenticated": true,
  "user": {
    "id": 12345,
    "login": "github_username",
    "name": "Full Name",
    "avatar_url": "https://avatars.githubusercontent.com/u/12345"
  }
}
```

---

## 8. Web Portal API

These endpoints support the web portal interface for agent management.

### Portal Home Page

- **URL**: `/portal`
- **Method**: `GET`
- **Description**: Renders the main portal dashboard
- **Authentication**: Requires GitHub authentication

### Agent Creation Page

- **URL**: `/portal/agents/create`
- **Method**: `GET`
- **Description**: Renders the agent creation interface
- **Authentication**: Requires GitHub authentication

### Agent Detail Page

- **URL**: `/portal/agents/{agent_id}`
- **Method**: `GET`
- **Description**: Renders the agent detail page with logs and status
- **Authentication**: Requires GitHub authentication

### Training Example Management

- **URL**: `/portal/agents/{agent_id}/training`
- **Method**: `GET`
- **Description**: Renders the training example management interface
- **Authentication**: Requires GitHub authentication

### Submit Training Examples

- **URL**: `/portal/agents/{agent_id}/training`
- **Method**: `POST`
- **Description**: Submits new training examples for the agent's interest model
- **Authentication**: Requires GitHub authentication

**Request Body**:
```json
{
  "examples": [
    {
      "content": "How do I upgrade my account?",
      "is_interested": true
    },
    {
      "content": "What's the weather like today?",
      "is_interested": false
    }
  ]
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Added 2 training examples",
  "training_id": "train_12345"
}
```

### Start Model Fine-Tuning

- **URL**: `/portal/agents/{agent_id}/fine-tune`
- **Method**: `POST`
- **Description**: Starts the fine-tuning process for the agent's interest model
- **Authentication**: Requires GitHub authentication

**Response**:
```json
{
  "status": "success",
  "message": "Fine-tuning started",
  "job_id": "ft_12345"
}
```

### Get Fine-Tuning Status

- **URL**: `/portal/agents/{agent_id}/fine-tune/{job_id}`
- **Method**: `GET`
- **Description**: Gets the status of a fine-tuning job
- **Authentication**: Requires GitHub authentication

**Response**:
```json
{
  "status": "completed",
  "progress": 100,
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.92,
    "recall": 0.94
  },
  "model_path": "/app/fine_tuned_model/model.pt"
}
```

---

## Error Handling

All API endpoints follow a standard error response format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Common HTTP status codes:

- `200 OK`: Request succeeded
- `400 Bad Request`: Invalid input parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server-side error
- `503 Service Unavailable`: Agent deployment service is unavailable

## Notes on Containerized Environment

In a containerized environment:

1. The core API runs on port 8888 inside the `core-system-1` container
2. The agent service API runs on port 8889 inside the `agent-service-1` container
3. Agents communicate with the core system via the core API
4. The event bus (Redis) is used for asynchronous communication
5. Authentication is managed through GitHub OAuth with secure token storage
6. Network communication is facilitated through the `semantic-subscription` Docker network
