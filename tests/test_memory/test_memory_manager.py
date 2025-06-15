#!/usr/bin/env python3
"""
Fixed Memory Manager Test
Tests basic functionality with proper module imports
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[0]}")

def test_settings_import():
    """Test that settings can be imported without errors"""
    print("🧪 Testing Settings Import...")
    
    try:
        # Mock environment variables to avoid external dependencies
        with patch.dict(os.environ, {}, clear=False):
            
            # Import the settings
            from config.settings import settings, Settings
            
            assert settings is not None
            assert isinstance(settings, Settings)
            assert settings.PROJECT_NAME == "Multi-Agent Data Science System"
            
        print("✅ Settings import successful")
        return True
        
    except Exception as e:
        print(f"❌ Settings import failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_manager_basic():
    """Test basic memory manager functionality"""
    print("🧪 Testing Memory Manager Basic Functionality...")
    
    try:
        # Create temporary directory for testing
        temp_dir = Path(tempfile.mkdtemp())
        print(f"  Using temp directory: {temp_dir}")
        
        # Mock all CAMEL dependencies to avoid external API calls
        camel_mocks = {}
        
        # Mock CAMEL imports
        camel_modules = [
            'camel.memories',
            'camel.storages', 
            'camel.embeddings',
            'camel.utils',
            'camel.types',
            'camel.messages'
        ]
        
        for module in camel_modules:
            camel_mocks[module] = MagicMock()
        
        with patch.dict('sys.modules', camel_mocks):
            
            # Setup specific mocks for the classes we use
            from unittest.mock import MagicMock
            
            # Mock OpenAIEmbedding
            mock_embedding = MagicMock()
            mock_embedding.get_output_dim.return_value = 1536
            camel_mocks['camel.embeddings'].OpenAIEmbedding.return_value = mock_embedding
            
            # Mock QdrantStorage
            mock_qdrant = MagicMock()
            camel_mocks['camel.storages'].QdrantStorage.return_value = mock_qdrant
            
            # Mock JsonStorage
            mock_json = MagicMock()
            camel_mocks['camel.storages'].JsonStorage.return_value = mock_json
            
            # Mock other CAMEL classes
            camel_mocks['camel.memories'].LongtermAgentMemory = MagicMock()
            camel_mocks['camel.memories'].ChatHistoryBlock = MagicMock()
            camel_mocks['camel.memories'].VectorDBBlock = MagicMock()
            camel_mocks['camel.memories'].MemoryRecord = MagicMock()
            camel_mocks['camel.memories'].ScoreBasedContextCreator = MagicMock()
            camel_mocks['camel.utils'].OpenAITokenCounter = MagicMock()
            camel_mocks['camel.types'].ModelType = MagicMock()
            camel_mocks['camel.types'].OpenAIBackendRole = MagicMock()
            camel_mocks['camel.messages'].BaseMessage = MagicMock()
            
            # Mock settings to use our temporary directory
            with patch('config.settings.settings') as mock_settings:
                mock_settings.MEMORY_PATH = temp_dir
                mock_settings.MEMORY_TOKEN_LIMIT = 1000
                mock_settings.VECTOR_DB_COLLECTION = "test_collection"
                
                # Now import and test memory manager
                from memory.memory_manager import MemoryManager, MemoryContext
                
                # Test 1: Basic initialization
                print("  ✓ Testing initialization...")
                mm = MemoryManager(max_context_tokens=500)
                assert mm.max_context_tokens == 500
                print("  ✓ Memory Manager initialized successfully")
                
                # Test 2: Memory context
                print("  ✓ Testing memory context...")
                context = MemoryContext(
                    agent_name="test_agent",
                    phase="data_cleaning",
                    task_id="test_task"
                )
                assert context.agent_name == "test_agent"
                assert context.phase == "data_cleaning"
                print("  ✓ Memory context works correctly")
                
                # Test 3: Agent memory creation (mock the method to avoid CAMEL calls)
                print("  ✓ Testing agent memory creation...")
                with patch.object(mm, '_create_memory_instance') as mock_create:
                    mock_memory = MagicMock()
                    mock_create.return_value = mock_memory
                    
                    agent_memory = mm.get_or_create_agent_memory("test_agent")
                    assert agent_memory is not None
                    assert "test_agent" in mm.agent_memories
                print("  ✓ Agent memory created successfully")
                
                # Test 4: Statistics
                print("  ✓ Testing memory statistics...")
                stats = mm.get_memory_statistics()
                assert isinstance(stats, dict)
                assert "agent_memories_count" in stats
                print("  ✓ Memory statistics work correctly")
                
                # Test 5: Shutdown
                print("  ✓ Testing shutdown...")
                mm.shutdown()
                assert len(mm.agent_memories) == 0
                print("  ✓ Shutdown works correctly")
                
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
                
        print("🎉 Memory manager tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_imports():
    """Test basic module imports"""
    print("🧪 Testing Basic Imports...")
    
    try:
        # Test that we can import required modules
        import config
        print("  ✓ config module imported")
        
        import memory
        print("  ✓ memory module imported")
        
        # Test specific imports
        from config import settings
        print("  ✓ settings imported from config")
        
        from memory import MemoryManager, MemoryContext
        print("  ✓ MemoryManager and MemoryContext imported from memory")
        
        print("✅ All basic imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Memory Manager Tests")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Basic imports
    if test_basic_imports():
        success_count += 1
    
    print()
    
    # Test 2: Settings import
    if test_settings_import():
        success_count += 1
    
    print()
    
    # Test 3: Memory manager functionality
    if test_memory_manager_basic():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {success_count}/{total_tests} passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Memory Manager is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())