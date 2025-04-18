import os
import logging
import asyncio
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from pydantic import SecretStr
import psutil
import uuid
from enum import Enum

from browser_use import Agent, BrowserConfig, Browser

# Configuração de logging
from browser_use.logging_config import setup_logging
setup_logging()
logger = logging.getLogger("browser-use-api")

# Carregar variáveis de ambiente
load_dotenv()

app = FastAPI(title="Browser-use API", description="API para controlar o Browser-use")

# Enum para status das tarefas
class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# Modelos de dados
class BrowserConfigModel(BaseModel):
    headless: bool = True
    disable_security: bool = True
    extra_chromium_args: List[str] = []

class ModelConfig(BaseModel):
    provider: str = Field(..., description="Provedor do modelo: openai, anthropic, google, ollama")
    model_name: str = Field(..., description="Nome do modelo a ser utilizado")
    api_key: Optional[str] = Field(None, description="API key para o provedor (se necessário)")
    azure_endpoint: Optional[str] = Field(None, description="Endpoint para Azure OpenAI (se provider=azure)")
    azure_api_version: Optional[str] = Field(None, description="Versão da API do Azure OpenAI (se provider=azure)")
    temperature: float = Field(0.0, description="Temperatura para geração (0.0 a 1.0)")

class TaskRequest(BaseModel):
    task: str
    llm_config: ModelConfig
    browser_config: Optional[BrowserConfigModel] = None
    max_steps: int = 20
    use_vision: bool = True

class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    result: Optional[str] = None
    error: Optional[str] = None
    steps_executed: Optional[int] = None

# Gerenciador de tarefas
class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.semaphore = asyncio.Semaphore(self._calculate_max_parallel_tasks())
    
    def _calculate_max_parallel_tasks(self) -> int:
        """Calcula o número máximo de tarefas paralelas baseado nos recursos da máquina"""
        cpu_count = psutil.cpu_count(logical=False) or 2  # Núcleos físicos
        memory_gb = psutil.virtual_memory().total / (1024**3)  # Memória em GB
        
        # Limite baseado em CPU (1 tarefa por núcleo)
        cpu_limit = cpu_count
        
        # Limite baseado em memória (1 tarefa por 2GB)
        memory_limit = int(memory_gb / 2)
        
        # Usar o menor dos dois limites
        return min(cpu_limit, memory_limit)
    
    async def create_task(self, request: TaskRequest) -> str:
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "status": TaskStatus.PENDING,
            "request": request,
            "result": None,
            "error": None,
            "steps_executed": 0
        }
        asyncio.create_task(self._execute_task(task_id))
        return task_id
    
    async def _execute_task(self, task_id: str):
        try:
            async with self.semaphore:
                task = self.tasks[task_id]
                task["status"] = TaskStatus.RUNNING
                
                try:
                    # Configurar o modelo LLM
                    llm = get_llm(task["request"].llm_config)
                    
                    # Configurar o navegador
                    browser_config = BrowserConfig(
                        headless=task["request"].browser_config.headless if task["request"].browser_config else True,
                        disable_security=task["request"].browser_config.disable_security if task["request"].browser_config else True,
                        extra_chromium_args=task["request"].browser_config.extra_chromium_args if task["request"].browser_config else []
                    )
                    
                    # Inicializar o navegador
                    browser = Browser(config=browser_config)
                    
                    # Inicializar e executar o agente
                    agent = Agent(
                        task=task["request"].task, 
                        llm=llm, 
                        browser=browser,
                        use_vision=task["request"].use_vision
                    )
                    
                    result = await agent.run(max_steps=task["request"].max_steps)
                    
                    # Extrair o resultado
                    success = False
                    content = "Tarefa não concluída"
                    
                    if result and result.history and len(result.history) > 0:
                        last_item = result.history[-1]
                        if last_item.result and len(last_item.result) > 0:
                            last_result = last_item.result[-1]
                            content = last_result.extracted_content or "Sem conteúdo extraído"
                            success = last_result.is_done
                    
                    task["result"] = content
                    task["status"] = TaskStatus.COMPLETED if success else TaskStatus.FAILED
                    task["steps_executed"] = len(result.history) if result.history else 0
                    
                    # Fechar o navegador após o uso
                    await browser.close()
                    
                except Exception as e:
                    task["status"] = TaskStatus.FAILED
                    task["error"] = str(e)
                    logger.error(f"Erro ao executar tarefa {task_id}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Erro ao executar tarefa {task_id}: {str(e)}")
            task["status"] = TaskStatus.FAILED
            task["error"] = str(e)
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        return self.tasks.get(task_id)

# Inicializar o gerenciador de tarefas
task_manager = TaskManager()

# Função para obter o LLM com base na configuração
def get_llm(model_config: ModelConfig):
    try:
        provider = model_config.provider.lower()
        
        if provider == "openai":
            return ChatOpenAI(
                model=model_config.model_name,
                temperature=model_config.temperature,
                api_key=model_config.api_key or os.getenv("OPENAI_API_KEY")
            )
        elif provider == "azure":
            return AzureChatOpenAI(
                model=model_config.model_name,
                temperature=model_config.temperature,
                api_key=SecretStr(model_config.api_key or os.getenv("AZURE_OPENAI_KEY", "")),
                azure_endpoint=model_config.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_version=model_config.azure_api_version or "2024-10-21"
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                model_name=model_config.model_name,
                temperature=model_config.temperature,
                api_key=model_config.api_key or os.getenv("ANTHROPIC_API_KEY")
            )
        elif provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_config.model_name,
                temperature=model_config.temperature,
                google_api_key=(model_config.api_key or os.getenv("GOOGLE_API_KEY", ""))
            )
        elif provider == "ollama":
            return ChatOllama(
                model=model_config.model_name,
                temperature=model_config.temperature
            )
        else:
            raise ValueError(f"Provedor não suportado: {provider}")
    except Exception as e:
        logger.error(f"Erro ao inicializar LLM: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao inicializar LLM: {str(e)}")

@app.post("/run", response_model=TaskResponse)
async def run_agent(request: TaskRequest = Body(...)):
    try:
        task_id = await task_manager.create_task(request)
        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="Tarefa criada com sucesso"
        )
    except Exception as e:
        logger.error(f"Erro ao criar tarefa: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/run-status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    task = task_manager.get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Tarefa não encontrada")
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        result=task["result"],
        error=task["error"],
        steps_executed=task["steps_executed"]
    )

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    
    # Obter porta do ambiente ou usar 8000 como padrão
    port = int(os.getenv("PORT", 8000))
    
    # Iniciar servidor
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info") 