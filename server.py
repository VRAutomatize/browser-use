import os
import logging
import asyncio
import traceback
import sys
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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
from datetime import datetime
import aiohttp
import json
from contextlib import asynccontextmanager

from browser_use import Agent, BrowserConfig, Browser

# Configuração de logging
from browser_use.logging_config import setup_logging
setup_logging()
logger = logging.getLogger("browser-use-api")

# Carregar variáveis de ambiente
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Iniciar o coletor de métricas
    metrics_task = asyncio.create_task(metrics_collector.start())
    yield
    # Parar o coletor de métricas e fechar o handler de erros
    await metrics_collector.stop()
    await error_handler.close()
    metrics_task.cancel()
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="Browser-use API",
    description="API para controlar o Browser-use",
    lifespan=lifespan
)

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
    elapsed_seconds: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

class SystemMetrics(BaseModel):
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    cpu_count: int
    max_parallel_tasks: int

class RunningTaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    start_time: datetime
    elapsed_seconds: float
    steps_executed: int

class SystemStatusResponse(BaseModel):
    system_metrics: SystemMetrics
    running_tasks: List[RunningTaskInfo]
    total_tasks: int
    pending_tasks: int
    running_tasks_count: int
    completed_tasks: int
    failed_tasks: int
    available_slots: int  # Número de vagas disponíveis para novas tarefas

# Classe para gerenciamento de erros
class ErrorHandler:
    def __init__(self):
        self.webhook_url = "https://vrautomatize-n8n.snrhk1.easypanel.host/webhook/browser-use-vra-handler"
        self.session = None
    
    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def notify_error(self, error: Exception, context: Dict[str, Any] = None):
        """Notifica o webhook sobre um erro"""
        try:
            session = await self._get_session()
            
            # Capturar stack trace
            stack_trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            
            # Preparar payload
            payload = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "stack_trace": stack_trace,
                "timestamp": datetime.now().isoformat(),
                "context": context or {},
                "system_info": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "cpu_count": psutil.cpu_count(),
                    "memory_usage": psutil.virtual_memory()._asdict(),
                    "disk_usage": psutil.disk_usage('/')._asdict()
                }
            }
            
            async with session.post(self.webhook_url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Erro ao notificar webhook de erro: {response.status} - {await response.text()}")
        except Exception as e:
            logger.error(f"Erro ao enviar notificação de erro: {str(e)}")
    
    async def close(self):
        """Fecha a sessão HTTP"""
        if self.session:
            await self.session.close()
            self.session = None

# Inicializar o handler de erros
error_handler = ErrorHandler()

# Middleware para capturar erros globais
@app.middleware("http")
async def error_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        # Capturar contexto da requisição
        context = {
            "path": request.url.path,
            "method": request.method,
            "headers": dict(request.headers),
            "query_params": dict(request.query_params),
            "client_host": request.client.host if request.client else None
        }
        
        # Notificar erro
        await error_handler.notify_error(e, context)
        
        # Retornar resposta de erro
        return JSONResponse(
            status_code=500,
            content={"detail": "Ocorreu um erro interno no servidor"}
        )

# Gerenciador de tarefas
class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.semaphore = asyncio.Semaphore(self._calculate_max_parallel_tasks())
        self.webhook_url = "https://vrautomatize-n8n.snrhk1.easypanel.host/webhook/notify-run"
        self.llm_config = None  # Configuração do LLM será definida na primeira tarefa
    
    def _calculate_max_parallel_tasks(self) -> int:
        """Calcula o número máximo de tarefas paralelas baseado nos recursos da máquina"""
        cpu_count = psutil.cpu_count(logical=False) or 2  # Núcleos físicos
        memory_gb = psutil.virtual_memory().total / (1024**3)  # Memória em GB
        
        # Limite baseado em CPU (2 tarefas por núcleo físico)
        cpu_limit = cpu_count * 2
        
        # Limite baseado em memória (1 tarefa por 1GB)
        memory_limit = int(memory_gb)
        
        # Usar o menor dos dois limites
        return min(cpu_limit, memory_limit)
    
    async def _notify_webhook(self, task_id: str, request: TaskRequest):
        """Notifica o webhook sobre uma nova tarefa"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "task_id": task_id,
                    "task": request.task,
                    "llm_config": {
                        "provider": request.llm_config.provider,
                        "model_name": request.llm_config.model_name,
                        "temperature": request.llm_config.temperature
                    },
                    "browser_config": {
                        "headless": request.browser_config.headless if request.browser_config else True,
                        "disable_security": request.browser_config.disable_security if request.browser_config else True
                    },
                    "max_steps": request.max_steps,
                    "use_vision": request.use_vision,
                    "timestamp": datetime.now().isoformat()
                }
                
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Erro ao notificar webhook: {response.status} - {await response.text()}")
        except Exception as e:
            await error_handler.notify_error(e, {"context": "notify_webhook", "task_id": task_id})
    
    async def create_task(self, request: TaskRequest) -> str:
        # Se for a primeira tarefa, armazena a configuração do LLM
        if self.llm_config is None:
            self.llm_config = request.llm_config
            
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "status": TaskStatus.PENDING,
            "request": request,
            "result": None,
            "error": None,
            "steps_executed": 0,
            "start_time": None,
            "end_time": None
        }
        
        # Notificar webhook em background
        asyncio.create_task(self._notify_webhook(task_id, request))
        
        # Iniciar execução da tarefa
        asyncio.create_task(self._execute_task(task_id))
        return task_id
    
    async def _execute_task(self, task_id: str):
        try:
            async with self.semaphore:
                task = self.tasks[task_id]
                task["status"] = TaskStatus.RUNNING
                task["start_time"] = datetime.now()
                
                # Executa a tarefa como um todo
                result, steps_count = await self._execute_command(
                    task["request"].task,
                    task["request"].browser_config,
                    task["request"].max_steps,
                    task["request"].use_vision
                )
                
                # Atualiza o status final e os passos executados
                task["status"] = TaskStatus.COMPLETED
                task["end_time"] = datetime.now()
                task["result"] = result
                task["steps_executed"] = steps_count
                
        except Exception as e:
            task["status"] = TaskStatus.FAILED
            task["error"] = str(e)
            task["end_time"] = datetime.now()
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        return self.tasks.get(task_id)
    
    def get_system_status(self) -> SystemStatusResponse:
        # Obter métricas do sistema
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        cpu_count = psutil.cpu_count(logical=False) or 2
        
        # Contar tarefas por status
        status_counts = {
            TaskStatus.PENDING: 0,
            TaskStatus.RUNNING: 0,
            TaskStatus.COMPLETED: 0,
            TaskStatus.FAILED: 0
        }
        
        running_tasks = []
        for task_id, task in self.tasks.items():
            status_counts[task["status"]] += 1
            
            if task["status"] == TaskStatus.RUNNING:
                elapsed_seconds = (datetime.now() - task["start_time"]).total_seconds() if task["start_time"] else 0
                running_tasks.append(RunningTaskInfo(
                    task_id=task_id,
                    status=task["status"],
                    start_time=task["start_time"],
                    elapsed_seconds=elapsed_seconds,
                    steps_executed=task["steps_executed"]
                ))
        
        # Calcular vagas disponíveis
        max_parallel_tasks = self._calculate_max_parallel_tasks()
        available_slots = max(0, max_parallel_tasks - status_counts[TaskStatus.RUNNING])
        
        return SystemStatusResponse(
            system_metrics=SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                cpu_count=cpu_count,
                max_parallel_tasks=max_parallel_tasks
            ),
            running_tasks=running_tasks,
            total_tasks=len(self.tasks),
            pending_tasks=status_counts[TaskStatus.PENDING],
            running_tasks_count=status_counts[TaskStatus.RUNNING],
            completed_tasks=status_counts[TaskStatus.COMPLETED],
            failed_tasks=status_counts[TaskStatus.FAILED],
            available_slots=available_slots
        )

    async def _execute_command(self, step: str, browser_config: Optional[BrowserConfigModel] = None, max_steps: int = 20, use_vision: bool = True) -> str:
        try:
            # Configurar o modelo LLM
            llm = get_llm(self.llm_config)
            
            # Configurar o navegador
            browser_config_obj = BrowserConfig(
                headless=browser_config.headless if browser_config else False,
                disable_security=browser_config.disable_security if browser_config else True,
                extra_chromium_args=browser_config.extra_chromium_args if browser_config else []
            )
            
            # Inicializar o navegador
            browser = Browser(config=browser_config_obj)
            
            # Inicializar e executar o agente
            agent = Agent(
                task=step,
                llm=llm,
                browser=browser,
                use_vision=use_vision
            )
            
            # Usar max_steps da requisição
            result = await agent.run(max_steps=max_steps)
            
            # Extrair o resultado final e contar os passos
            content = "Passo não concluído"
            steps_count = 0
            
            if result and result.history:
                # Contar passos e extrair resultado
                for item in result.history:
                    if item.result:
                        steps_count += 1
                        # Verificar se é o resultado final
                        if isinstance(item.result, str):
                            content = item.result
                        elif hasattr(item.result, 'done'):
                            content = item.result.done.get('text', 'Sem resultado')
                        elif hasattr(item.result, 'text'):
                            content = item.result.text
            
            # Fechar o navegador após o uso
            await browser.close()
            
            return content, steps_count
            
        except Exception as e:
            await error_handler.notify_error(e, {
                "context": "execute_command",
                "step": step
            })
            return f"Erro ao executar passo: {str(e)}", 0

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
    
    elapsed_seconds = None
    if task["start_time"]:
        if task["end_time"]:
            elapsed_seconds = (task["end_time"] - task["start_time"]).total_seconds()
        else:
            elapsed_seconds = (datetime.now() - task["start_time"]).total_seconds()
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        result=task["result"],
        error=task["error"],
        steps_executed=task["steps_executed"],
        elapsed_seconds=elapsed_seconds,
        start_time=task["start_time"],
        end_time=task["end_time"]
    )

@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    return task_manager.get_system_status()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Classe para gerenciamento de métricas periódicas
class MetricsCollector:
    def __init__(self):
        self.webhook_url = "https://vrautomatize-n8n.snrhk1.easypanel.host/webhook/status"
        self.session = None
        self.is_running = False
    
    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def collect_metrics(self):
        """Coleta métricas do sistema e tarefas"""
        try:
            session = await self._get_session()
            
            # Obter métricas do sistema
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Obter métricas das tarefas
            status_counts = {
                TaskStatus.PENDING: 0,
                TaskStatus.RUNNING: 0,
                TaskStatus.COMPLETED: 0,
                TaskStatus.FAILED: 0
            }
            
            running_tasks = []
            for task_id, task in task_manager.tasks.items():
                status_counts[task["status"]] += 1
                
                if task["status"] == TaskStatus.RUNNING:
                    elapsed_seconds = (datetime.now() - task["start_time"]).total_seconds() if task["start_time"] else 0
                    running_tasks.append({
                        "task_id": task_id,
                        "status": task["status"],
                        "start_time": task["start_time"].isoformat() if task["start_time"] else None,
                        "elapsed_seconds": elapsed_seconds,
                        "steps_executed": task["steps_executed"]
                    })
            
            # Calcular vagas disponíveis
            max_parallel_tasks = task_manager._calculate_max_parallel_tasks()
            available_slots = max(0, max_parallel_tasks - status_counts[TaskStatus.RUNNING])
            
            # Preparar payload
            payload = {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_total_gb": memory.total / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_used_gb": disk.used / (1024**3),
                    "disk_total_gb": disk.total / (1024**3),
                    "cpu_count": psutil.cpu_count(logical=False) or 2,
                    "max_parallel_tasks": max_parallel_tasks,
                    "available_slots": available_slots
                },
                "task_metrics": {
                    "total_tasks": len(task_manager.tasks),
                    "pending_tasks": status_counts[TaskStatus.PENDING],
                    "running_tasks": status_counts[TaskStatus.RUNNING],
                    "completed_tasks": status_counts[TaskStatus.COMPLETED],
                    "failed_tasks": status_counts[TaskStatus.FAILED],
                    "running_tasks_details": running_tasks
                }
            }
            
            # Enviar métricas para o webhook
            async with session.post(self.webhook_url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Erro ao enviar métricas: {response.status} - {await response.text()}")
        except Exception as e:
            logger.error(f"Erro ao coletar métricas: {str(e)}")
            await error_handler.notify_error(e, {"context": "metrics_collection"})
    
    async def start(self):
        """Inicia a coleta periódica de métricas"""
        if not self.is_running:
            self.is_running = True
            while self.is_running:
                try:
                    await self.collect_metrics()
                except Exception as e:
                    logger.error(f"Erro no loop de métricas: {str(e)}")
                await asyncio.sleep(600)  # 10 minutos
    
    async def stop(self):
        """Para a coleta de métricas"""
        self.is_running = False
        if self.session:
            await self.session.close()
            self.session = None

# Inicializar o coletor de métricas
metrics_collector = MetricsCollector()

if __name__ == "__main__":
    import uvicorn
    
    # Obter porta do ambiente ou usar 8000 como padrão
    port = int(os.getenv("PORT", 8000))
    
    # Iniciar servidor
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info") 