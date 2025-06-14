"""
Advanced Configuration System for Multi-Agent Data Science Platform
Supports HIPAA/SOC2 compliance, auto-configuration, and flexible user customization
"""

import os
import secrets
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import yaml
import structlog
from pydantic import BaseSettings, Field, validator, SecretStr
from pydantic_settings import SettingsConfigDict

# Setup structured logging with colorama
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)

class LogLevel(str, Enum):
    """Logging levels with color coding"""
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ModelProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"
    CUSTOM = "custom"

class DatabaseType(str, Enum):
    """Supported database types"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    BIGQUERY = "bigquery"

class AuthMethod(str, Enum):
    """Authentication methods"""
    JWT = "jwt"
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    BASIC = "basic"

class Settings(BaseSettings):
    """
    Comprehensive settings class supporting auto-configuration and user customization
    HIPAA/SOC2 compliant with encryption and secure defaults
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow"  # Allow additional fields for user customization
    )
    
    # === Core Application Settings ===
    PROJECT_NAME: str = "Multi-Agent Data Science System"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Advanced AI-powered data science automation platform"
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    ENVIRONMENT: str = Field(default="development", description="Environment: development, staging, production")
    
    # === Security & Compliance (HIPAA/SOC2) ===
    SECRET_KEY: SecretStr = Field(default_factory=lambda: SecretStr(secrets.token_urlsafe(32)))
    ENCRYPTION_KEY: SecretStr = Field(default_factory=lambda: SecretStr(secrets.token_urlsafe(32)))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="JWT token expiry")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="Refresh token expiry")
    PASSWORD_MIN_LENGTH: int = Field(default=12, description="Minimum password length")
    SESSION_TIMEOUT_MINUTES: int = Field(default=60, description="Session timeout")
    MAX_LOGIN_ATTEMPTS: int = Field(default=5, description="Maximum login attempts before lockout")
    
    # === API Configuration ===
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, description="API port")
    API_PREFIX: str = Field(default="/api/v1", description="API path prefix")
    CORS_ORIGINS: List[str] = Field(default=["http://localhost:3000", "http://localhost:8501"], description="CORS origins")
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, description="API rate limit per minute")
    
    # === Authentication ===
    AUTH_METHOD: AuthMethod = Field(default=AuthMethod.JWT, description="Authentication method")
    ENABLE_USER_REGISTRATION: bool = Field(default=True, description="Allow user registration")
    REQUIRE_EMAIL_VERIFICATION: bool = Field(default=True, description="Require email verification")
    
    # === LLM Provider Configuration (User Configurable) ===
    DEFAULT_MODEL_PROVIDER: ModelProvider = Field(default=ModelProvider.OPENAI, description="Default LLM provider")
    DEFAULT_MODEL_NAME: str = Field(default="gpt-4o-mini", description="Default model name")
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[SecretStr] = Field(default=None, description="OpenAI API key")
    OPENAI_API_BASE: Optional[str] = Field(default=None, description="Custom OpenAI API base URL")
    OPENAI_ORGANIZATION: Optional[str] = Field(default=None, description="OpenAI organization ID")
    
    # Anthropic Configuration
    ANTHROPIC_API_KEY: Optional[SecretStr] = Field(default=None, description="Anthropic API key")
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY: Optional[SecretStr] = Field(default=None, description="Azure OpenAI API key")
    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(default=None, description="Azure OpenAI endpoint")
    AZURE_OPENAI_API_VERSION: str = Field(default="2024-02-01", description="Azure OpenAI API version")
    
    # Google Configuration
    GOOGLE_API_KEY: Optional[SecretStr] = Field(default=None, description="Google API key")
    GOOGLE_PROJECT_ID: Optional[str] = Field(default=None, description="Google Cloud project ID")
    
    # === Database Configuration ===
    DATABASE_TYPE: DatabaseType = Field(default=DatabaseType.POSTGRESQL, description="Database type")
    
    # PostgreSQL (Default)
    POSTGRES_HOST: str = Field(default="localhost", description="PostgreSQL host")
    POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL port")
    POSTGRES_USER: str = Field(default="ds_admin", description="PostgreSQL username")
    POSTGRES_PASSWORD: SecretStr = Field(default=SecretStr("ds_secure_2024!"), description="PostgreSQL password")
    POSTGRES_DB: str = Field(default="multi_agent_ds", description="PostgreSQL database name")
    
    # Connection pooling
    DATABASE_POOL_SIZE: int = Field(default=10, description="Database connection pool size")
    DATABASE_MAX_OVERFLOW: int = Field(default=20, description="Database max overflow connections")
    DATABASE_POOL_TIMEOUT: int = Field(default=30, description="Database pool timeout seconds")
    
    # === Vector Database (Qdrant) ===
    QDRANT_HOST: str = Field(default="localhost", description="Qdrant host")
    QDRANT_PORT: int = Field(default=6333, description="Qdrant port")
    QDRANT_API_KEY: Optional[SecretStr] = Field(default=None, description="Qdrant API key")
    QDRANT_HTTPS: bool = Field(default=False, description="Use HTTPS for Qdrant")
    
    # === Redis Configuration ===
    REDIS_HOST: str = Field(default="localhost", description="Redis host")
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    REDIS_PASSWORD: Optional[SecretStr] = Field(default=None, description="Redis password")
    REDIS_DB: int = Field(default=0, description="Redis database number")
    
    # === Storage Paths ===
    BASE_DIR: Path = Field(default_factory=lambda: Path.cwd(), description="Base directory")
    STORAGE_PATH: Path = Field(default_factory=lambda: Path("./storage"), description="Storage path")
    MEMORY_PATH: Path = Field(default_factory=lambda: Path("./storage/memory"), description="Memory storage path")
    MODELS_PATH: Path = Field(default_factory=lambda: Path("./storage/models"), description="Models storage path")
    LOGS_PATH: Path = Field(default_factory=lambda: Path("./storage/logs"), description="Logs storage path")
    DATA_PATH: Path = Field(default_factory=lambda: Path("./data"), description="Data storage path")
    REPORTS_PATH: Path = Field(default_factory=lambda: Path("./data/outputs/reports"), description="Reports output path")
    
    # === Logging Configuration ===
    LOG_LEVEL: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    LOG_FORMAT: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    LOG_TO_FILE: bool = Field(default=True, description="Enable file logging")
    LOG_ROTATION: str = Field(default="midnight", description="Log rotation schedule")
    LOG_RETENTION_DAYS: int = Field(default=30, description="Log retention in days")
    ENABLE_STRUCTURED_LOGGING: bool = Field(default=True, description="Enable structured logging")
    
    # === Memory & Context Configuration ===
    MEMORY_TOKEN_LIMIT: int = Field(default=8192, description="Memory context token limit")
    VECTOR_DB_COLLECTION: str = Field(default="data_science_knowledge", description="Vector DB collection name")
    CHAT_HISTORY_LIMIT: int = Field(default=1000, description="Chat history message limit")
    MEMORY_CLEANUP_INTERVAL_HOURS: int = Field(default=24, description="Memory cleanup interval")
    
    # === Workflow Configuration ===
    MAX_TASK_RETRIES: int = Field(default=3, description="Maximum task retry attempts")
    TASK_TIMEOUT_MINUTES: int = Field(default=30, description="Task execution timeout")
    QA_REVIEW_ENABLED: bool = Field(default=True, description="Enable QA review process")
    PARALLEL_TASK_LIMIT: int = Field(default=5, description="Maximum parallel tasks")
    WORKFLOW_SAVE_INTERVAL_MINUTES: int = Field(default=5, description="Workflow state save interval")
    
    # === Agent Configuration ===
    AGENT_RESPONSE_TIMEOUT: int = Field(default=300, description="Agent response timeout seconds")
    AGENT_MAX_ITERATIONS: int = Field(default=10, description="Maximum agent iterations per task")
    ENABLE_AGENT_MEMORY: bool = Field(default=True, description="Enable agent memory")
    AGENT_TEMPERATURE: float = Field(default=0.1, description="Default agent temperature")
    
    # === Data Processing Configuration ===
    MAX_FILE_SIZE_MB: int = Field(default=100, description="Maximum file size in MB")
    SUPPORTED_FILE_TYPES: List[str] = Field(
        default=["csv", "xlsx", "json", "parquet", "sql"],
        description="Supported file types"
    )
    DATA_VALIDATION_ENABLED: bool = Field(default=True, description="Enable data validation")
    AUTO_DATA_PROFILING: bool = Field(default=True, description="Enable automatic data profiling")
    
    # === Visualization Configuration ===
    DEFAULT_PLOT_STYLE: str = Field(default="seaborn-v0_8", description="Default matplotlib style")
    PLOT_DPI: int = Field(default=300, description="Plot resolution DPI")
    PLOT_FORMAT: str = Field(default="png", description="Default plot format")
    INTERACTIVE_PLOTS: bool = Field(default=True, description="Enable interactive plots")
    
    # === Cloud Storage Configuration ===
    AWS_ACCESS_KEY_ID: Optional[SecretStr] = Field(default=None, description="AWS access key")
    AWS_SECRET_ACCESS_KEY: Optional[SecretStr] = Field(default=None, description="AWS secret key")
    AWS_REGION: str = Field(default="us-east-1", description="AWS region")
    AWS_S3_BUCKET: Optional[str] = Field(default=None, description="AWS S3 bucket name")
    
    GOOGLE_CLOUD_PROJECT: Optional[str] = Field(default=None, description="Google Cloud project")
    GOOGLE_CREDENTIALS_PATH: Optional[Path] = Field(default=None, description="Google credentials file path")
    
    # === Monitoring & Health Checks ===
    HEALTH_CHECK_INTERVAL: int = Field(default=60, description="Health check interval seconds")
    ENABLE_METRICS: bool = Field(default=True, description="Enable metrics collection")
    METRICS_PORT: int = Field(default=8080, description="Metrics server port")
    
    # === User Customization Support ===
    CUSTOM_AGENT_CONFIGS: Dict[str, Any] = Field(default_factory=dict, description="Custom agent configurations")
    CUSTOM_WORKFLOW_CONFIGS: Dict[str, Any] = Field(default_factory=dict, description="Custom workflow configurations")
    CUSTOM_TOOL_CONFIGS: Dict[str, Any] = Field(default_factory=dict, description="Custom tool configurations")
    
    @validator("STORAGE_PATH", "MEMORY_PATH", "MODELS_PATH", "LOGS_PATH", "DATA_PATH", "REPORTS_PATH")
    def create_directories(cls, v):
        """Ensure directories exist"""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level"""
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v
    
    @property
    def database_url(self) -> str:
        """Generate database URL based on configuration"""
        if self.DATABASE_TYPE == DatabaseType.POSTGRESQL:
            password = self.POSTGRES_PASSWORD.get_secret_value()
            return f"postgresql://{self.POSTGRES_USER}:{password}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        # Add other database types as needed
        raise ValueError(f"Unsupported database type: {self.DATABASE_TYPE}")
    
    @property
    def async_database_url(self) -> str:
        """Generate async database URL"""
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://")
    
    @property
    def redis_url(self) -> str:
        """Generate Redis URL"""
        password_part = f":{self.REDIS_PASSWORD.get_secret_value()}@" if self.REDIS_PASSWORD else ""
        return f"redis://{password_part}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    def get_model_config(self, provider: Optional[ModelProvider] = None) -> Dict[str, Any]:
        """Get model configuration for specified provider"""
        provider = provider or self.DEFAULT_MODEL_PROVIDER
        
        base_config = {
            "temperature": self.AGENT_TEMPERATURE,
            "max_tokens": 4096,
            "timeout": self.AGENT_RESPONSE_TIMEOUT
        }
        
        if provider == ModelProvider.OPENAI:
            return {
                **base_config,
                "api_key": self.OPENAI_API_KEY.get_secret_value() if self.OPENAI_API_KEY else None,
                "api_base": self.OPENAI_API_BASE,
                "organization": self.OPENAI_ORGANIZATION,
                "model": self.DEFAULT_MODEL_NAME
            }
        elif provider == ModelProvider.ANTHROPIC:
            return {
                **base_config,
                "api_key": self.ANTHROPIC_API_KEY.get_secret_value() if self.ANTHROPIC_API_KEY else None,
                "model": "claude-3-5-sonnet-latest"
            }
        # Add other providers as needed
        
        return base_config
    
    def setup_logging(self):
        """Setup structured logging with color coding"""
        import logging
        from rich.logging import RichHandler
        import sys
        
        # Color mapping for log levels
        level_colors = {
            "DEBUG": Fore.CYAN,
            "INFO": Fore.GREEN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "CRITICAL": Fore.MAGENTA + Style.BRIGHT
        }
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.LOG_LEVEL.value))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler with rich formatting
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_level=True,
            show_path=True
        )
        console_handler.setLevel(getattr(logging, self.LOG_LEVEL.value))
        root_logger.addHandler(console_handler)
        
        # File handler if enabled
        if self.LOG_TO_FILE:
            from logging.handlers import TimedRotatingFileHandler
            
            log_file = self.LOGS_PATH / "application.log"
            file_handler = TimedRotatingFileHandler(
                log_file,
                when=self.LOG_ROTATION,
                backupCount=self.LOG_RETENTION_DAYS
            )
            file_handler.setLevel(getattr(logging, self.LOG_LEVEL.value))
            
            formatter = logging.Formatter(self.LOG_FORMAT)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Setup structlog if enabled
        if self.ENABLE_STRUCTURED_LOGGING:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
    
    class Config:
        """Pydantic configuration"""
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"

def load_custom_configs() -> Dict[str, Any]:
    """Load custom configurations from YAML files"""
    configs = {}
    config_dir = Path(__file__).parent
    
    config_files = [
        "agent_configs.yaml",
        "workflow_configs.yaml", 
        "database_configs.yaml"
    ]
    
    for config_file in config_files:
        config_path = config_dir / config_file
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    configs[config_file.replace('.yaml', '')] = yaml.safe_load(f)
            except Exception as e:
                print(f"{Fore.YELLOW}Warning: Could not load {config_file}: {e}")
    
    return configs

def get_settings() -> Settings:
    """Get configured settings instance"""
    _settings = Settings()
    _settings.setup_logging()
    return _settings

# Global settings instance
settings = get_settings()

# Load custom configurations
custom_configs = load_custom_configs()

# Export commonly used settings
__all__ = [
    "Settings",
    "settings", 
    "custom_configs",
    "LogLevel",
    "ModelProvider", 
    "DatabaseType",
    "AuthMethod",
    "get_settings",
    "load_custom_configs"
]