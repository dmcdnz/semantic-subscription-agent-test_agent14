# Fine-Tuned Model Interest Subscription System

## Overview

The Semantic Subscription System employs a unique approach to distributed message processing through fine-tuned model interest subscription. Unlike traditional message routing systems that use explicit topics or queues, our system uses semantic similarity to determine which agents should process which messages. This document details how this approach works in the containerized environment.

## Core Concepts

### Semantic Subscription

Semantic subscription refers to the process of subscribing to messages based on their semantic meaning rather than explicit routing rules. This is implemented using vector embeddings, where:

1. Each message is converted into a vector representation (embedding)
2. Each agent has its own set of interest vectors (from fine-tuned models)
3. Agents "subscribe" to messages whose vectors are similar to their interest vectors

### Two-Phase Processing

The system operates in two distinct phases:

1. **Interest Registration Phase**: Determines which agents are interested in a message
2. **Processing Phase**: Interested agents process the message and create responses

### Event-Driven Architecture

The system uses an event-driven architecture where:

1. Events are published to a distributed event bus (Redis)
2. Agents subscribe to specific event types
3. The system operates asynchronously without central orchestration

## Key Components

### Core Components

1. **Vector Database (Qdrant)** - Stores message embeddings and enables similarity search
2. **Event Bus (Redis)** - Enables distributed event-driven communication
3. **Agent Interest Service** - Determines which agents are interested in each message
4. **Message Processing Service** - Manages the processing of messages by interested agents
5. **Agent Manager** - Tracks and manages all available agents

### Agent Components

1. **Fine-Tuned Models** - Domain-specific models for accurate interest determination
2. **Interest Model** - Uses embeddings to calculate similarity scores
3. **Agent Base Class** - Provides common functionality for all agents
4. **Custom Agent Implementations** - Domain-specific processing logic

## Containerization Architecture

The system is containerized using Docker Compose with the following services:

### Infrastructure Services

1. **Redis Service (`redis`)** 
   - Provides the backbone for the distributed event bus
   - Enables communication between containers
   - Persists data using a named volume (`redis_data`)
   - Exposes port 6379 for client connections
   - Includes health checks to ensure availability

2. **Qdrant Service (`qdrant`)**
   - Provides the vector database for semantic search
   - Stores message embeddings and enables similarity search
   - Persists data using a named volume (`qdrant_data`)
   - Exposes ports 6333 (HTTP API) and 6334 (GRPC)
   - Includes health checks to verify API availability

### Application Services

3. **Core System (`core-system-1`)**
   - Built from `Dockerfile.core` with a multi-stage build process
   - Runs the central event-driven system (`run_events.py`)
   - Configured with environment variables:  
     - `USE_REDIS_EVENT_BUS=true` - Uses Redis-backed event bus
     - `VECTOR_DB_BACKEND=qdrant` - Uses Qdrant for vector storage
     - Connection details for Redis and Qdrant services
   - Mounts the Docker socket to manage containers
   - Exposes port 8888 for the API and web interface
   - Depends on the Redis and Qdrant services

4. **Agent Service (`agent-service-1`)**
   - Built from `Dockerfile.agent-service`
   - Responsible for:
     - GitHub repository operations for agent code
     - Building and deploying agent containers
     - Managing agent lifecycle (start/stop/update)
     - Monitoring agent health and status
   - Includes Docker CLI to create and manage agent containers
   - Exposes port 8889 for agent management API
   - Waits for the core system to be healthy before starting

5. **Individual Agent Containers**
   - Dynamically created by the Agent Service
   - Each agent runs in its own isolated container
   - Built from agent-specific repositories
   - Connect to the Redis event bus and Qdrant vector database
   - Use fine-tuned models from their local directories

## Data Flow

### Message Ingestion Flow

1. A new message is added to the `MessageRepository`
2. The repository generates an embedding for the message
3. The message and its embedding are stored in the vector database
4. A `message.created` event is published to the event bus

### Interest Registration Flow

1. The `AgentInterestService` receives the `message.created` event
2. It retrieves all registered agents from the `AgentManager`
3. For each agent, it calls `calculate_interest(message)` to determine interest level
4. Agents with interest scores above their threshold are marked as interested
5. The message is updated with the list of interested agents
6. A `message.interest.registered` event is published with the message and interested agents

### Message Processing Flow

1. Interested agents receive the `message.interest.registered` event
2. Each agent processes the message using its domain-specific logic
3. Agent responses are added to the `MessageRepository` as new messages
4. These new messages can trigger additional interest and processing

## Key Scripts and their Functions

### Containerized System Components

#### `core-system-1` Container

Main container that runs the central event-driven system.

**Key Responsibilities:**
- Manages the vector database (Qdrant)
- Runs the distributed event bus (Redis)
- Handles interest registration and message processing
- Provides the API endpoints for agent and system interaction

**Container Configuration:** Defined in `docker-compose.yml` and built from `Dockerfile.core`

#### `agent-service-1` Container

Manages agent deployment and lifecycle.

**Key Responsibilities:**
- Handles GitHub repository operations
- Builds and deploys agent containers
- Manages agent lifecycle (start/stop/update)
- Monitors agent health and status

**Container Configuration:** Defined in `docker-compose.yml` and built from `Dockerfile.agent-service`

#### `run_events.py`

Event-driven entry point used by the core-system container.

**Key Functions:**
- Initializes the event-driven architecture components
- Sets up Redis-backed event bus and vector database connections
- Registers core event handlers

### Core Module Scripts

#### `semsubscription/core/event_bus.py`

Implements the in-memory event bus for development and testing environments.

**Key Classes:**
- `EventBus`: Central event bus for publish-subscribe patterns

**Key Methods:**
- `subscribe(event_type, callback)`: Subscribe to an event type
- `publish(event_type, data)`: Publish an event to all subscribers
- `unsubscribe(event_type, callback)`: Unsubscribe from an event type
- `_safe_execute(callback, data)`: Execute callbacks safely with error handling

**Usage Context:**
- Used in single-process environments (non-containerized)
- Supports thread-safe operations within a single process
- Lower latency but limited to single-machine deployments

#### `semsubscription/core/distributed_event_bus.py`

Implements the Redis-backed distributed event bus for containerized environments.

**Key Classes:**
- `RedisEventBus`: Redis-based implementation of the event bus

**Key Methods:**
- `subscribe(event_type, callback)`: Subscribe to an event type
- `publish(event_type, data)`: Publish an event with JSON serialization
- `unsubscribe(event_type, callback)`: Unsubscribe from an event type
- `_get_prefixed_channel(event_type)`: Adds user prefixes for multi-user support
- `_message_handler()`: Background thread for processing Redis messages
- `_json_serializer(obj)`: Custom JSON serialization for complex objects

**Usage Context:**
- Used in the containerized environment (docker-compose)
- Enables cross-container communication
- Supports multi-user deployment through channel prefixing
- Core component for distributed event-driven architecture

#### `semsubscription/core/event_bus_factory.py`

Factory pattern for selecting the appropriate event bus implementation.

**Key Functions:**
- `get_event_bus()`: Returns appropriate event bus based on configuration

#### `semsubscription/core/agent_interest_service.py`

Service that implements Phase 1 (interest registration).

**Key Classes:**
- `AgentInterestService`: Manages interest registration for all agents

**Key Methods:**
- `determine_interested_agents(message)`: Determines which agents are interested in a message

**Inputs:** Message object
**Outputs:** Updates message with interested agents and publishes event

#### `semsubscription/core/message_repository.py`

Manages all messages in the system.

**Key Classes:**
- `MessageRepository`: Central repository for managing messages

**Key Methods:**
- `add_message(content, metadata)`: Adds a new message to the repository
- `add_response(original_message_id, content, agent_id, metadata)`: Adds a response to a message
- `search_similar(query, limit, threshold)`: Searches for similar messages

**Inputs:** Message content and metadata
**Outputs:** Stored messages, responses, search results

### Vector Database Scripts

#### `semsubscription/vector_db/database.py`

Defines the interface for vector database operations.

**Key Classes:**
- `Message`: Base class for message data
- `BaseVectorDatabase`: Abstract base class for vector database implementations

#### `semsubscription/vector_db/embedding.py`

Handles embedding generation for messages and interest modeling.

**Key Classes:**
- `EmbeddingEngine`: Generates vector embeddings for text
- `InterestModel`: Represents an agent's interests using embeddings

**Key Methods:**
- `embed_text(text)`: Generates embeddings for text
- `calculate_similarity(text)`: Calculates similarity to interest vectors
- `train(examples, method)`: Trains the interest model with examples

**Inputs:** Text to embed, training examples
**Outputs:** Vector embeddings, similarity scores

#### `semsubscription/vector_db/backends/qdrant_db.py`

Qdrant implementation of the vector database.

**Key Classes:**
- `QdrantVectorDatabase`: Qdrant-based implementation

**Key Methods:**
- `add_message(message, embedding)`: Adds a message with its embedding
- `search_similar(query_vector, k, threshold)`: Searches for similar messages

**Inputs:** Messages, query vectors
**Outputs:** Stored messages, search results

### Agent-Related Scripts

#### `semsubscription/agents/base_enhanced.py`

Enhanced base agent implementation.

**Key Classes:**
- `EnhancedAgent`: Base class for all agents
- `InterestClassificationHead`: Neural network for interest classification

**Key Methods:**
- `calculate_interest(message)`: Calculates interest score for a message
- `is_interested(message)`: Determines if the agent is interested in a message
- `process_message(message)`: Processes a message (to be implemented by subclasses)

**Inputs:** Messages
**Outputs:** Interest scores, processing results

### Agent Management Scripts

#### `scripts/create_agent_repo.py`

Creates a new agent repository on GitHub.

**Key Functions:**
- `create_agent_repo(name, domain, description, github_token)`: Creates GitHub repo from template

**Inputs:** Agent name, domain, description, GitHub token
**Outputs:** GitHub repository URL

#### `scripts/create_agent_container.py`

Creates a Docker container for an agent.

**Key Functions:**
- `create_agent(config_path, build_container)`: Creates agent from configuration

**Inputs:** Config path, build flag
**Outputs:** Container image

## Fine-Tuned Model Integration

### Model Training

Fine-tuned models are created through the following process:

1. Example messages relevant to the agent's domain are collected
2. A base embedding model (typically Sentence-BERT/MiniLM) is fine-tuned on these examples
3. The fine-tuned model is saved to the agent's `fine_tuned_model` directory

### Model Usage in Containerized Environment

In the containerized environment:

1. During agent initialization, the `setup_interest_model()` method checks for a fine-tuned model
2. If available, it creates an `EmbeddingEngine` pointing to the local model
3. This engine is used to create an `InterestModel` for semantic interest determination
4. If no fine-tuned model is found, it falls back to the default model and interest determination

### Agent Template Integration

The agent templates include:

1. `agent.py.template`: Core agent implementation with fine-tuned model support
2. `interest_model.py.template`: Custom interest model with domain-specific enhancements
3. `embedding_engine.py.template`: Embedding engine for containerized environments

These templates ensure that newly created agents can use fine-tuned models without requiring the full system dependencies.

## Agent Build and Deployment Process

### GitHub Integration

The system uses GitHub as the primary code storage and collaboration platform for agents:

1. **Code Storage:** Agent code is stored in GitHub repositories with standardized structure
2. **Template-Based:** New agents are created from templates via the `create_agent_repo.py` script
3. **Versioning:** GitHub provides version control for tracking changes to agent code
4. **Collaboration:** Multiple developers can collaborate on agent implementations

### Server-Side Build Process

Unlike traditional CI/CD with GitHub Actions, the system uses a server-side build approach:

1. **Repository Cloning:** The `agent-service-1` container clones the agent repository
2. **Docker Build:** A custom Docker image is built for the agent using its Dockerfile
3. **Fine-Tuned Model Integration:** Models are copied into the container during build
4. **Container Deployment:** The container is launched and connected to the event bus

### Agent Creation Workflow

1. Developer runs `scripts/create_agent_repo.py` to create a new agent repository
2. The script:
   - Creates a new GitHub repository with standardized structure
   - Populates template files (agent.py, interest_model.py, embedding_engine.py)
   - Adds a directory structure for fine-tuned models
   - Creates a Dockerfile for containerization
3. Developer adds domain-specific code and fine-tuned models
4. The agent is deployed through the agent management web portal

### Fine-Tuned Model Integration

Fine-tuned models are a core component of the semantic subscription approach:

1. **Model Development:** Models are fine-tuned using domain-specific examples
2. **Model Storage:** The fine-tuned model is stored in the agent's repository in the `fine_tuned_model` directory
3. **Containerization:** During container building, the model is packaged with the agent
4. **Runtime Loading:** The agent's `setup_interest_model()` method loads the model at startup
5. **Fallback Mechanism:** If a fine-tuned model is not available, the system falls back to the default model

## Authentication and User Management

The system includes robust GitHub authentication and user management features:

1. **OAuth Integration:** Secure GitHub OAuth flow with PKCE for user authentication
2. **Token Management:** Secure storage and handling of GitHub access tokens
3. **User Isolation:** Logical separation of resources and data between users
4. **Repository Access Control:** Users can only deploy agents from repositories they have access to
5. **Session Management:** Secure cookie-based session handling for the web portal

## Advanced Containerization Features

### Network Configuration

Containerized agents communicate with core services through a dedicated Docker network:

1. **Network Isolation:** Agents run in the `semantic-subscription` network
2. **Host Communication:** Proper configuration of `host.docker.internal` and `extra_hosts`
3. **Service Discovery:** Containers resolve service names to appropriate endpoints
4. **Port Mapping:** Strategic port mapping for external and internal communications

### Fallback Mechanisms

The system includes several fallback mechanisms for resilience:

1. **Interest Model Fallback:** If a fine-tuned model is unavailable, agents fall back to default models
2. **Polling Fallback:** Agents can fall back to polling mode if the event bus is unavailable
3. **Template Compatibility:** Support for both `{variable}` and `{{VARIABLE}}` template formats
4. **Repository Caching:** Caching mechanisms for repository data in case of token issues

### Health Monitoring

Containerized deployment includes comprehensive health monitoring:

1. **Container Status Checks:** Regular checks of container running status
2. **Log Aggregation:** Capture and display of agent logs through the web portal
3. **Event Monitoring:** Tracking of agent participation in the event flow
4. **Startup Sequence Validation:** Ensuring proper service initialization order

### Web Portal

The web portal provides a comprehensive interface for agent management:

1. **Agent Creation:** Interface for creating new agent repositories
2. **Deployment Management:** Controls for building, starting, and stopping agents
3. **Training Interface:** Tools for generating and managing training examples
4. **Model Fine-Tuning:** Interface for fine-tuning interest models
5. **Status Monitoring:** Real-time tracking of agent status and activity

## Conclusion

The Fine-Tuned Model Interest Subscription system provides a powerful framework for semantic message routing and decentralized processing. By leveraging vector embeddings and domain-specific fine-tuned models, it enables emergent behavior and flexible processing patterns without requiring explicit workflow definitions.

The containerized implementation enhances this architecture with scalability, isolation, and efficient deployment, while maintaining the core semantic subscription approach.
