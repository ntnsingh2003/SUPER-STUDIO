from backend.agents.synthetic_data_agent import synthetic_data_agent
from backend.agents.hallucination_agent import hallucination_agent
from backend.agents.creative_agent import creative_agent
from backend.agents.codebase_agent import codebase_agent
from backend.agents.marketing_agent import marketing_agent
from backend.agents.image_gen_agent import image_gen_agent

def route_to_agent(service: str, prompt: str) -> str:
    """
    Routes a user prompt to the specific agent based on the requested service.
    """
    if service == "synthetic-data":
        return synthetic_data_agent(prompt)
    
    elif service == "hallucination-auditor":
        return hallucination_agent(prompt)
    
    elif service == "creative":
        return creative_agent(prompt)
    
    elif service == "codebase":
        return codebase_agent(prompt)
    
    elif service == "marketing":
        return marketing_agent(prompt)
    
    elif service == "image-gen":
        return image_gen_agent(prompt)
    
    else:
        return f"Error: Service '{service}' not recognized by the AI Orchestrator."
