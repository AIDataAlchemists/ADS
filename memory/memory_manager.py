"""
Memory Manager for Multi-Agent Data Science System
Handles memory orchestration across agents with CAMEL framework integration
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import structlog

from camel.memories import (
    LongtermAgentMemory,
    ChatHistoryBlock, 
    VectorDBBlock,
    MemoryRecord,
    ScoreBasedContextCreator
)
from camel.storages import QdrantStorage, JsonStorage
from camel.embeddings import OpenAIEmbedding
from camel.utils import OpenAITokenCounter
from camel.types import ModelType, OpenAIBackendRole
from camel.messages import BaseMessage

from config.settings import settings

logger = structlog.get_logger(__name__)

@dataclass
class MemoryContext:
    """Context information for memory operations"""
    agent_name: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    phase: Optional[str] = None
    task_id: Optional[str] = None

class MemoryManager:
    """
    Memory Manager for orchestrating memory across all agents in the system
    
    Features:
    - Agent-specific memory instances
    - Global shared memory for cross-agent knowledge
    - Phase-specific memory for workflow continuity
    - CAMEL framework integration
    """
    
    def __init__(self, max_context_tokens: int = None):
        """
        Initialize the Memory Manager
        
        Args:
            max_context_tokens: Maximum tokens for context generation
        """
        self.max_context_tokens = max_context_tokens or settings.MEMORY_TOKEN_LIMIT
        
        # Memory instances for different agents
        self.agent_memories: Dict[str, LongtermAgentMemory] = {}
        
        # Global shared memory for cross-agent knowledge sharing
        self.global_memory: Optional[LongtermAgentMemory] = None
        
        # Phase-specific memories for workflow continuity
        self.phase_memories: Dict[str, LongtermAgentMemory] = {}
        
        # Initialize core memory systems
        self._setup_memory_systems()
        
        logger.info(
            "Memory Manager initialized",
            max_context_tokens=self.max_context_tokens,
            storage_path=str(settings.MEMORY_PATH)
        )
    
    def _setup_memory_systems(self):
        """Setup core memory systems"""
        try:
            # Ensure memory directory exists
            settings.MEMORY_PATH.mkdir(parents=True, exist_ok=True)
            
            # Setup global shared memory
            self.global_memory = self._create_memory_instance("global_shared")
            
            # Setup phase memories for the 7-phase workflow
            phases = [
                "research_development",
                "data_collection", 
                "data_cleaning",
                "data_preprocessing",
                "eda",
                "feature_engineering",
                "model_development",
                "presentation"
            ]
            
            for phase in phases:
                self.phase_memories[phase] = self._create_memory_instance(f"phase_{phase}")
            
            logger.info("Memory systems initialized successfully")
            
        except Exception as e:
            logger.error("Failed to setup memory systems", error=str(e))
            raise RuntimeError(f"Memory system initialization failed: {str(e)}")
    
    def _create_memory_instance(self, instance_name: str) -> LongtermAgentMemory:
        """Create a new memory instance with proper CAMEL configuration"""
        
        try:
            # Setup vector storage
            collection_name = f"{settings.VECTOR_DB_COLLECTION}_{instance_name}"
            vector_storage = QdrantStorage(
                vector_dim=OpenAIEmbedding().get_output_dim(),
                collection_name=collection_name,
                path=str(settings.MEMORY_PATH / f"{instance_name}_vector_db")
            )
            
            # Setup chat history storage
            chat_storage_path = settings.MEMORY_PATH / f"{instance_name}_chat_history.json"
            chat_storage = JsonStorage(chat_storage_path)
            
            # Create memory blocks
            chat_history = ChatHistoryBlock(storage=chat_storage)
            vector_db = VectorDBBlock(
                storage=vector_storage,
                embedding=OpenAIEmbedding()
            )
            
            # Create context creator
            context_creator = ScoreBasedContextCreator(
                token_counter=OpenAITokenCounter(ModelType.GPT_4O_MINI),
                token_limit=self.max_context_tokens
            )
            
            # Initialize memory instance
            memory_instance = LongtermAgentMemory(
                context_creator=context_creator,
                chat_history_block=chat_history,
                vector_db_block=vector_db
            )
            
            logger.debug(f"Created memory instance: {instance_name}")
            return memory_instance
            
        except Exception as e:
            logger.error(
                "Failed to create memory instance", 
                instance_name=instance_name,
                error=str(e)
            )
            raise RuntimeError(f"Memory instance creation failed for {instance_name}: {str(e)}")
    
    def get_or_create_agent_memory(self, agent_name: str) -> LongtermAgentMemory:
        """Get or create memory instance for a specific agent"""
        
        if agent_name not in self.agent_memories:
            self.agent_memories[agent_name] = self._create_memory_instance(f"agent_{agent_name}")
            logger.info(f"Created memory for agent: {agent_name}")
        
        return self.agent_memories[agent_name]
    
    def store_interaction(
        self,
        context: MemoryContext,
        user_message: BaseMessage,
        assistant_message: BaseMessage
    ) -> bool:
        """
        Store agent interaction in appropriate memory systems
        
        Args:
            context: Memory context with agent and phase information
            user_message: User's message
            assistant_message: Assistant's response
            
        Returns:
            bool: True if successful, False otherwise
        """
        
        try:
            # Create memory records
            user_record = MemoryRecord(
                message=user_message,
                role_at_backend=OpenAIBackendRole.USER
            )
            
            assistant_record = MemoryRecord(
                message=assistant_message,
                role_at_backend=OpenAIBackendRole.ASSISTANT
            )
            
            records = [user_record, assistant_record]
            
            # Store in agent-specific memory
            agent_memory = self.get_or_create_agent_memory(context.agent_name)
            agent_memory.write_records(records)
            
            # Store in global shared memory for cross-agent knowledge
            if self.global_memory:
                self.global_memory.write_records(records)
            
            # Store in phase-specific memory if phase is specified
            if context.phase and context.phase in self.phase_memories:
                self.phase_memories[context.phase].write_records(records)
            
            logger.info(
                "Interaction stored successfully",
                agent_name=context.agent_name,
                phase=context.phase,
                user_content_length=len(user_message.content),
                assistant_content_length=len(assistant_message.content)
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to store interaction",
                agent_name=context.agent_name,
                error=str(e)
            )
            return False
    
    def get_context(
        self,
        context: MemoryContext,
        include_global: bool = True,
        include_phase: bool = True
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get contextual memory for an agent
        
        Args:
            context: Memory context with agent information
            include_global: Whether to include global shared memory
            include_phase: Whether to include phase-specific memory
            
        Returns:
            Tuple of (context_messages, total_token_count)
        """
        
        try:
            all_contexts = []
            total_tokens = 0
            
            # Get agent-specific memory
            agent_memory = self.get_or_create_agent_memory(context.agent_name)
            agent_context, agent_tokens = agent_memory.get_context()
            if agent_context:
                all_contexts.extend(agent_context)
                total_tokens += agent_tokens
            
            # Get phase-specific memory if requested and available
            if include_phase and context.phase and context.phase in self.phase_memories:
                phase_memory = self.phase_memories[context.phase]
                phase_context, phase_tokens = phase_memory.get_context()
                if phase_context:
                    all_contexts.extend(phase_context)
                    total_tokens += phase_tokens
            
            # Get global memory if requested (limit to avoid token overflow)
            if include_global and self.global_memory and total_tokens < self.max_context_tokens:
                remaining_tokens = self.max_context_tokens - total_tokens
                if remaining_tokens > 100:  # Only if substantial tokens remain
                    global_context, global_tokens = self.global_memory.get_context()
                    if global_context:
                        # Take only what fits in remaining token budget
                        all_contexts.extend(global_context)
                        total_tokens += min(global_tokens, remaining_tokens)
            
            logger.debug(
                "Context retrieved",
                agent_name=context.agent_name,
                phase=context.phase,
                total_contexts=len(all_contexts),
                total_tokens=total_tokens
            )
            
            return all_contexts, total_tokens
            
        except Exception as e:
            logger.error(
                "Failed to retrieve context",
                agent_name=context.agent_name,
                error=str(e)
            )
            return [], 0
    
    def clear_agent_memory(self, agent_name: str) -> bool:
        """Clear memory for a specific agent"""
        
        try:
            if agent_name in self.agent_memories:
                self.agent_memories[agent_name].clear()
                logger.info(f"Cleared memory for agent: {agent_name}")
                return True
            else:
                logger.warning(f"No memory found for agent: {agent_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to clear memory for agent {agent_name}", error=str(e))
            return False
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        
        try:
            stats = {
                "agent_memories_count": len(self.agent_memories),
                "phase_memories_count": len(self.phase_memories),
                "global_memory_active": self.global_memory is not None,
                "max_context_tokens": self.max_context_tokens,
                "agents": list(self.agent_memories.keys()),
                "phases": list(self.phase_memories.keys())
            }
            
            # Try to get context sizes for agents
            agent_context_sizes = {}
            for agent_name in self.agent_memories:
                try:
                    context = MemoryContext(agent_name=agent_name)
                    _, token_count = self.get_context(context, include_global=False, include_phase=False)
                    agent_context_sizes[agent_name] = token_count
                except:
                    agent_context_sizes[agent_name] = 0
            
            stats["agent_context_sizes"] = agent_context_sizes
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get memory statistics", error=str(e))
            return {"error": str(e)}
    
    def shutdown(self):
        """Gracefully shutdown the memory manager"""
        
        try:
            logger.info("Shutting down Memory Manager")
            
            # Clear in-memory references
            self.agent_memories.clear()
            self.phase_memories.clear()
            self.global_memory = None
            
            logger.info("Memory Manager shutdown completed")
            
        except Exception as e:
            logger.error("Error during memory manager shutdown", error=str(e))