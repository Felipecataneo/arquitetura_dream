from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import asyncio
from dataclasses import asdict
from enum import Enum
from typing import Any, Optional, Dict
from dotenv import load_dotenv
import os
import time
from concurrent.futures import ThreadPoolExecutor
import uuid

# --- CARREGAR VARIÁVEIS DE AMBIENTE ---
load_dotenv()

# Importe a sua classe principal do seu arquivo
from arquiteto_final import DreamSystemV13_2, LLMClient

# Configurar logging
logging.basicConfig(level=logging.ERROR)
log = logging.getLogger('uvicorn')

app = FastAPI(title="DREAM V13.2 API", version="1.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic para validação
class ProblemRequest(BaseModel):
    problem: str
    model: Optional[str] = None

class AsyncProblemRequest(BaseModel):
    problem: str
    model: Optional[str] = None
    callback_url: Optional[str] = None  # URL para notificar quando terminar

class TaskStatus(BaseModel):
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None

# Armazenamento em memória para tarefas (em produção, use Redis/DB)
task_storage: Dict[str, TaskStatus] = {}

# Thread pool para execução assíncrona
thread_pool = ThreadPoolExecutor(max_workers=4)

# Inicialização global do agente
agent = None

@app.on_event("startup")
async def startup_event():
    global agent
    print("🧠 Initializing DREAM V12.3 System...")
    try:
        agent = DreamSystemV13_2()
        if agent.clients:
            print("✅ DREAM System Initialized Successfully.")
            print(f"   Available models: {list(agent.clients.keys())}")
        else:
            print("🚨 FATAL ERROR: No LLM clients could be initialized.")
            agent = None
    except Exception as e:
        print(f"🚨 FATAL ERROR during initialization: {e}")
        agent = None

@app.on_event("shutdown")
async def shutdown_event():
    thread_pool.shutdown(wait=True)

def convert_enums_to_strings(data: Any) -> Any:
    """Converte enums para strings para serialização JSON"""
    if isinstance(data, dict):
        return {key: convert_enums_to_strings(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_enums_to_strings(item) for item in data]
    elif isinstance(data, Enum):
        return data.value
    else:
        return data

def solve_problem_sync(problem: str, model_name: Optional[str], task_id: str) -> None:
    """Função síncrona para resolver problema (executada em thread separada)"""
    try:
        print(f"\n🔄 Processing task {task_id}: '{problem}' for model: '{model_name or 'default'}'")
        
        # Atualizar status para "processing"
        task_storage[task_id].status = "processing"
        
        # Resolver o problema
        cognitive_state = agent.solve_problem(problem, model_name=model_name)
        
        # Converter para dicionário e serializar
        response_dict = asdict(cognitive_state)
        serializable_response = convert_enums_to_strings(response_dict)
        
        # Atualizar com resultado
        task_storage[task_id].status = "completed"
        task_storage[task_id].result = serializable_response
        task_storage[task_id].completed_at = time.time()
        
        print(f"✅ Task {task_id} completed successfully. Strategy: {serializable_response.get('strategy', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Task {task_id} failed: {e}")
        task_storage[task_id].status = "failed"
        task_storage[task_id].error = str(e)
        task_storage[task_id].completed_at = time.time()

@app.get("/")
async def root():
    """Endpoint raiz para verificar se a API está funcionando"""
    return {"message": "DREAM V13.2 API is running", "status": "healthy"}

@app.get("/available_models")
async def get_available_models():
    """Retorna os modelos disponíveis"""
    if not agent:
        raise HTTPException(status_code=500, detail="DREAM system is not available.")
    
    available_models = list(agent.clients.keys())
    return available_models

@app.post("/solve")
async def solve(request: ProblemRequest):
    """Resolve um problema usando o sistema DREAM (síncrono - compatibilidade)"""
    if not agent:
        raise HTTPException(
            status_code=500, 
            detail="DREAM system is not available due to an initialization error."
        )
    
    if not request.problem:
        raise HTTPException(status_code=400, detail="No 'problem' field provided")
    
    try:
        print(f"\n🔄 Received problem: '{request.problem}' for model: '{request.model or 'default'}'")
        
        # Executar em thread separada para não bloquear
        loop = asyncio.get_event_loop()
        cognitive_state = await loop.run_in_executor(
            thread_pool, 
            agent.solve_problem, 
            request.problem, 
            request.model
        )
        
        # Converter para dicionário e serializar
        response_dict = asdict(cognitive_state)
        serializable_response = convert_enums_to_strings(response_dict)
        
        print(f"✅ Responded successfully. Strategy: {serializable_response.get('strategy', 'N/A')}")
        
        return serializable_response
        
    except Exception as e:
        logging.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An internal error occurred: {str(e)}"
        )

@app.post("/solve_async")
async def solve_async(request: AsyncProblemRequest):
    """Inicia resolução assíncrona de problema - retorna task_id imediatamente"""
    if not agent:
        raise HTTPException(
            status_code=500, 
            detail="DREAM system is not available due to an initialization error."
        )
    
    if not request.problem:
        raise HTTPException(status_code=400, detail="No 'problem' field provided")
    
    # Gerar ID único para a tarefa
    task_id = str(uuid.uuid4())
    
    # Criar entrada no armazenamento
    task_storage[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        created_at=time.time()
    )
    
    # Executar em background
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        thread_pool, 
        solve_problem_sync, 
        request.problem, 
        request.model, 
        task_id
    )
    
    return {"task_id": task_id, "status": "pending", "message": "Task started"}

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Verifica o status de uma tarefa assíncrona"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = task_storage[task_id]
    return {
        "task_id": task.task_id,
        "status": task.status,
        "result": task.result,
        "error": task.error,
        "created_at": task.created_at,
        "completed_at": task.completed_at
    }

@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """Remove uma tarefa do armazenamento"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    del task_storage[task_id]
    return {"message": "Task deleted successfully"}

@app.get("/tasks")
async def list_tasks():
    """Lista todas as tarefas"""
    return {
        "tasks": [
            {
                "task_id": task.task_id,
                "status": task.status,
                "created_at": task.created_at,
                "completed_at": task.completed_at
            }
            for task in task_storage.values()
        ]
    }

# Para compatibilidade com Vercel
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI server for DREAM V12.3...")
    print("   API Endpoints available at:")
    print("   - http://127.0.0.1:8000/solve (POST)")
    print("   - http://127.0.0.1:8000/available_models (GET)")
    uvicorn.run(app, host="0.0.0.0", port=8000)