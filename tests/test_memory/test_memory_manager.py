#!/usr/bin/env python3
"""
Unit Tests for Memory Manager
Tests all core functionality with proper setup and teardown
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from camel.messages import BaseMessage
from camel.types import RoleType

# Import our memory manager
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from memory.memory_manager import MemoryManager, MemoryContext
from config.settings import settings

class TestMemoryManager:
    """Test suite for MemoryManager"""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test"""
        # Setup: Create temporary directory for testing
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Mock settings to use temporary directory
        with patch.object(settings, 'MEMORY_PATH', self.temp_dir):
            with patch.object(settings, 'MEMORY_TOKEN_LIMIT', 1000):
                with patch.object(settings, 'VECTOR_DB_COLLECTION', 'test_collection'):
                    yield
        
        # Teardown: Clean up temporary directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @pytest.fixture
    def mock_openai_embedding(self):
        """Mock OpenAI embedding to avoid API calls during testing"""
        with patch('memory.memory_manager.OpenAIEmbedding') as mock_embedding:
            mock_instance = MagicMock()
            mock_instance.get_output_dim.return_value = 1536  # Standard OpenAI embedding dimension
            mock_embedding.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def mock_qdrant_storage(self):
        """Mock Qdrant storage to avoid external dependencies"""
        with patch('memory.memory_manager.QdrantStorage') as mock_storage:
            mock_instance = MagicMock()
            mock_storage.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def memory_manager(self, mock_openai_embedding, mock_qdrant_storage):
        """Create a memory manager instance for testing"""
        return MemoryManager(max_context_tokens=500)
    
    def test_memory_manager_initialization(self, memory_manager):
        """Test that memory manager initializes correctly"""
        assert memory_manager is not None
        assert memory_manager.max_context_tokens == 500
        assert memory_manager.global_memory is not None
        assert len(memory_manager.phase_memories) == 8  # 7 phases + 1 extra
        assert memory_manager.agent_memories == {}  # Should be empty initially
    
    def test_get_or_create_agent_memory(self, memory_manager):
        """Test creating agent memory instances"""
        agent_name = "test_agent"
        
        # First call should create new memory
        memory1 = memory_manager.get_or_create_agent_memory(agent_name)
        assert memory1 is not None
        assert agent_name in memory_manager.agent_memories
        
        # Second call should return the same memory
        memory2 = memory_manager.get_or_create_agent_memory(agent_name)
        assert memory1 is memory2
        
        # Different agent should get different memory
        memory3 = memory_manager.get_or_create_agent_memory("different_agent")
        assert memory3 is not memory1
    
    def test_store_interaction_success(self, memory_manager):
        """Test storing agent interaction successfully"""
        # Create test context and messages
        context = MemoryContext(
            agent_name="test_agent",
            phase="data_cleaning",
            task_id="task_123"
        )
        
        user_message = BaseMessage.make_user_message(
            role_name="User",
            content="What is the best way to clean data?"
        )
        
        assistant_message = BaseMessage.make_assistant_message(
            role_name="DataCleaner",
            content="The best way to clean data is to start with identifying missing values and outliers."
        )
        
        # Mock the memory write operations to avoid CAMEL framework calls
        with patch.object(memory_manager.global_memory, 'write_records') as mock_global_write:
            with patch.object(memory_manager.phase_memories['data_cleaning'], 'write_records') as mock_phase_write:
                # Mock agent memory creation and write
                mock_agent_memory = MagicMock()
                with patch.object(memory_manager, 'get_or_create_agent_memory', return_value=mock_agent_memory):
                    
                    # Test the store operation
                    result = memory_manager.store_interaction(context, user_message, assistant_message)
                    
                    # Verify success
                    assert result is True
                    
                    # Verify all memory systems were called
                    mock_agent_memory.write_records.assert_called_once()
                    mock_global_write.assert_called_once()
                    mock_phase_write.assert_called_once()
    
    def test_store_interaction_failure(self, memory_manager):
        """Test handling of store interaction failure"""
        context = MemoryContext(agent_name="test_agent")
        
        user_message = BaseMessage.make_user_message(
            role_name="User",
            content="Test message"
        )
        
        assistant_message = BaseMessage.make_assistant_message(
            role_name="Assistant",
            content="Test response"
        )
        
        # Mock agent memory to raise an exception
        mock_agent_memory = MagicMock()
        mock_agent_memory.write_records.side_effect = Exception("Memory write failed")
        
        with patch.object(memory_manager, 'get_or_create_agent_memory', return_value=mock_agent_memory):
            result = memory_manager.store_interaction(context, user_message, assistant_message)
            
            # Should return False on failure
            assert result is False
    
    def test_get_context_agent_only(self, memory_manager):
        """Test getting context for agent-only memory"""
        context = MemoryContext(agent_name="test_agent")
        
        # Mock agent memory to return test context
        mock_agent_memory = MagicMock()
        test_context = [{"role": "user", "content": "test message"}]
        mock_agent_memory.get_context.return_value = (test_context, 50)
        
        with patch.object(memory_manager, 'get_or_create_agent_memory', return_value=mock_agent_memory):
            contexts, token_count = memory_manager.get_context(context, include_global=False, include_phase=False)
            
            assert contexts == test_context
            assert token_count == 50
            mock_agent_memory.get_context.assert_called_once()
    
    def test_get_context_with_phase_and_global(self, memory_manager):
        """Test getting context including phase and global memory"""
        context = MemoryContext(agent_name="test_agent", phase="data_cleaning")
        
        # Mock all memory systems
        mock_agent_memory = MagicMock()
        mock_agent_memory.get_context.return_value = ([{"content": "agent"}], 100)
        
        mock_phase_memory = memory_manager.phase_memories['data_cleaning']
        mock_phase_memory.get_context.return_value = ([{"content": "phase"}], 150)
        
        mock_global_memory = memory_manager.global_memory
        mock_global_memory.get_context.return_value = ([{"content": "global"}], 200)
        
        with patch.object(memory_manager, 'get_or_create_agent_memory', return_value=mock_agent_memory):
            contexts, token_count = memory_manager.get_context(context, include_global=True, include_phase=True)
            
            # Should include all three sources
            assert len(contexts) == 3
            assert token_count == 450  # 100 + 150 + 200
    
    def test_clear_agent_memory_success(self, memory_manager):
        """Test clearing agent memory successfully"""
        agent_name = "test_agent"
        
        # Create agent memory first
        mock_agent_memory = MagicMock()
        memory_manager.agent_memories[agent_name] = mock_agent_memory
        
        # Clear the memory
        result = memory_manager.clear_agent_memory(agent_name)
        
        assert result is True
        mock_agent_memory.clear.assert_called_once()
    
    def test_clear_agent_memory_not_found(self, memory_manager):
        """Test clearing memory for non-existent agent"""
        result = memory_manager.clear_agent_memory("non_existent_agent")
        assert result is False
    
    def test_get_memory_statistics(self, memory_manager):
        """Test getting memory statistics"""
        # Add some test agents
        memory_manager.get_or_create_agent_memory("agent1")
        memory_manager.get_or_create_agent_memory("agent2")
        
        stats = memory_manager.get_memory_statistics()
        
        assert isinstance(stats, dict)
        assert stats["agent_memories_count"] == 2
        assert stats["phase_memories_count"] == 8
        assert stats["global_memory_active"] is True
        assert stats["max_context_tokens"] == 500
        assert "agent1" in stats["agents"]
        assert "agent2" in stats["agents"]
        assert "data_cleaning" in stats["phases"]
    
    def test_shutdown(self, memory_manager):
        """Test graceful shutdown"""
        # Add some test data
        memory_manager.get_or_create_agent_memory("test_agent")
        
        # Verify data exists before shutdown
        assert len(memory_manager.agent_memories) == 1
        assert memory_manager.global_memory is not None
        
        # Shutdown
        memory_manager.shutdown()
        
        # Verify cleanup
        assert len(memory_manager.agent_memories) == 0
        assert len(memory_manager.phase_memories) == 0
        assert memory_manager.global_memory is None


def run_manual_test():
    """Manual test function to run outside pytest"""
    print("🧪 Running Manual Memory Manager Tests...")
    
    try:
        # Test 1: Basic initialization
        print("\n1️⃣ Testing Memory Manager Initialization...")
        with patch('memory.memory_manager.OpenAIEmbedding'), \
             patch('memory.memory_manager.QdrantStorage'), \
             patch.object(settings, 'MEMORY_PATH', Path('./test_memory')), \
             patch.object(settings, 'MEMORY_TOKEN_LIMIT', 1000), \
             patch.object(settings, 'VECTOR_DB_COLLECTION', 'test'):
            
            mm = MemoryManager(max_context_tokens=500)
            assert mm.max_context_tokens == 500
            print("✅ Memory Manager initialized successfully")
        
        # Test 2: Agent memory creation
        print("\n2️⃣ Testing Agent Memory Creation...")
        with patch('memory.memory_manager.OpenAIEmbedding'), \
             patch('memory.memory_manager.QdrantStorage'), \
             patch.object(settings, 'MEMORY_PATH', Path('./test_memory')), \
             patch.object(settings, 'MEMORY_TOKEN_LIMIT', 1000), \
             patch.object(settings, 'VECTOR_DB_COLLECTION', 'test'):
            
            mm = MemoryManager()
            agent_memory = mm.get_or_create_agent_memory("test_agent")
            assert agent_memory is not None
            assert "test_agent" in mm.agent_memories
            print("✅ Agent memory created successfully")
        
        # Test 3: Memory context
        print("\n3️⃣ Testing Memory Context...")
        context = MemoryContext(
            agent_name="test_agent",
            phase="data_cleaning",
            task_id="test_task"
        )
        assert context.agent_name == "test_agent"
        assert context.phase == "data_cleaning"
        print("✅ Memory context works correctly")
        
        # Test 4: Statistics
        print("\n4️⃣ Testing Memory Statistics...")
        with patch('memory.memory_manager.OpenAIEmbedding'), \
             patch('memory.memory_manager.QdrantStorage'), \
             patch.object(settings, 'MEMORY_PATH', Path('./test_memory')), \
             patch.object(settings, 'MEMORY_TOKEN_LIMIT', 1000), \
             patch.object(settings, 'VECTOR_DB_COLLECTION', 'test'):
            
            mm = MemoryManager()
            mm.get_or_create_agent_memory("agent1")
            mm.get_or_create_agent_memory("agent2")
            
            stats = mm.get_memory_statistics()
            assert stats["agent_memories_count"] == 2
            assert "agent1" in stats["agents"]
            print("✅ Memory statistics work correctly")
        
        print("\n🎉 All manual tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run manual tests when script is executed directly
    success = run_manual_test()
    exit(0 if success else 1)