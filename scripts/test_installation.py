#!/usr/bin/env python3
"""
Installation Test Script for Multi-Agent Data Science System
Tests all critical dependencies and CAMEL framework functionality
"""

import sys
import importlib
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

def test_import(module_name, display_name=None, version_attr=None):
    """Test importing a module and optionally check its version"""
    display_name = display_name or module_name
    try:
        module = importlib.import_module(module_name)
        version = ""
        if version_attr:
            try:
                version = f" v{getattr(module, version_attr)}"
            except AttributeError:
                version = " (version unknown)"
        
        print(f"{Fore.GREEN}✅ {display_name}{version}")
        return True
    except ImportError as e:
        print(f"{Fore.RED}❌ {display_name} - {str(e)}")
        return False

def test_camel_functionality():
    """Test basic CAMEL framework functionality"""
    try:
        from camel.agents import ChatAgent
        from camel.messages import BaseMessage
        from camel.types import ModelType, ModelPlatformType
        
        # Test basic message creation
        message = BaseMessage.make_user_message(
            role_name="test_user",
            content="Hello, this is a test message"
        )
        
        print(f"{Fore.GREEN}✅ CAMEL basic functionality works")
        return True
    except Exception as e:
        print(f"{Fore.RED}❌ CAMEL functionality test failed: {str(e)}")
        return False

def test_data_science_stack():
    """Test data science libraries"""
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create simple test data
        data = pd.DataFrame({
            'x': np.random.randn(10),
            'y': np.random.randn(10)
        })
        
        print(f"{Fore.GREEN}✅ Data science stack works")
        return True
    except Exception as e:
        print(f"{Fore.RED}❌ Data science stack test failed: {str(e)}")
        return False

def main():
    """Run all installation tests"""
    print(f"{Fore.CYAN}{Style.BRIGHT}🧪 Multi-Agent Data Science System - Installation Test")
    print(f"{Fore.CYAN}Python Version: {sys.version}")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 0
    
    # Core Python libraries
    print(f"\n{Fore.YELLOW}📦 Core Dependencies:")
    dependencies = [
        ("camel", "CAMEL-AI Framework", "__version__"),
        ("pandas", "Pandas", "__version__"),
        ("numpy", "NumPy", "__version__"),
        ("matplotlib", "Matplotlib", "__version__"),
        ("seaborn", "Seaborn", "__version__"),
        ("scikit-learn", "Scikit-Learn", "__version__"),
        ("fastapi", "FastAPI", "__version__"),
        ("pydantic", "Pydantic", "__version__"),
        ("sqlalchemy", "SQLAlchemy", "__version__"),
        ("qdrant_client", "Qdrant Client", "__version__"),
        ("redis", "Redis", "__version__"),
        ("openai", "OpenAI", "__version__"),
        ("anthropic", "Anthropic", "__version__"),
        ("colorama", "Colorama", "__version__"),
        ("structlog", "StructLog", "__version__"),
    ]
    
    for module, name, version_attr in dependencies:
        if test_import(module, name, version_attr):
            tests_passed += 1
        total_tests += 1
    
    # Functionality tests
    print(f"\n{Fore.YELLOW}⚙️  Functionality Tests:")
    
    # Test CAMEL functionality
    total_tests += 1
    if test_camel_functionality():
        tests_passed += 1
    
    # Test data science stack
    total_tests += 1
    if test_data_science_stack():
        tests_passed += 1
    
    # Summary
    print(f"\n{Fore.CYAN}📊 Test Summary:")
    print("=" * 60)
    
    success_rate = (tests_passed / total_tests) * 100
    
    if tests_passed == total_tests:
        print(f"{Fore.GREEN}🎉 All tests passed! ({tests_passed}/{total_tests})")
        print(f"{Fore.GREEN}✨ System is ready for development!")
    elif success_rate >= 80:
        print(f"{Fore.YELLOW}⚠️  Most tests passed ({tests_passed}/{total_tests}) - {success_rate:.1f}%")
        print(f"{Fore.YELLOW}Some optional dependencies may be missing")
    else:
        print(f"{Fore.RED}❌ Critical failures detected ({tests_passed}/{total_tests}) - {success_rate:.1f}%")
        print(f"{Fore.RED}Please check the failed imports above")
        return 1
    
    # Next steps
    print(f"\n{Fore.CYAN}🚀 Next Steps:")
    print("1. Copy .env.example to .env and configure your API keys")
    print("2. Ensure PostgreSQL and Qdrant are running")
    print("3. Run the core system components")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())