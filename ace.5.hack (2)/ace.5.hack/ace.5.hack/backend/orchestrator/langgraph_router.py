"""
LangGraph-Style Agent Orchestrator for AI Super Studio
Implements a state-machine workflow: Planner -> Agent(s) -> Evaluator
Supports true multi-agent pipeline chaining: each agent receives  
the previous agent's output as context.
"""
import time
from typing import Optional
from backend.agents.planner_agent import planner_agent
from backend.agents.evaluator_agent import evaluator_agent
from backend.agents.prompt_optimizer_agent import prompt_optimizer_agent
from backend.agents.research_agent import research_agent
from backend.agents.creative_agent import creative_agent
from backend.agents.codebase_agent import codebase_agent
from backend.agents.marketing_agent import marketing_agent
from backend.agents.hallucination_agent import hallucination_agent
from backend.agents.synthetic_data_agent import synthetic_data_agent
from backend.agents.image_gen_agent import image_gen_agent
from backend.memory.memory_store import add_message, get_context_string

# ── Agent Executor Registry ─────────────────────────────────────────
AGENT_EXECUTORS = {
    "creative":              creative_agent,
    "codebase":              codebase_agent,
    "marketing":             marketing_agent,
    "hallucination-auditor": hallucination_agent,
    "synthetic-data":        synthetic_data_agent,
    "image-gen":             image_gen_agent,
    "research":              research_agent,
    "prompt-optimizer":      lambda prompt, **kw: prompt_optimizer_agent(prompt),
}


def execute_workflow(
    prompt: str,
    session_id: str = "default",
    service: Optional[str] = None,
    use_planner: bool = True,
    optimize_prompt: bool = False
) -> dict:
    """
    Main orchestration entry point. LangGraph-style state machine:
    
    1. [Optional] Prompt Optimizer refines the user request
    2. Planner Agent determines which agents to chain (up to 3)
    3. Agents execute in sequence — each receives the previous output as context
    4. [Optional] Evaluator Agent validates the final output
    5. Results are stored in conversation memory
    """
    start_time = time.time()
    workflow_log = []

    # ── Store user message in memory ────────────────────────────────
    add_message(session_id, "user", prompt)
    context = get_context_string(session_id)

    # ── Step 1: Prompt Optimization (optional) ──────────────────────
    active_prompt = prompt
    if optimize_prompt:
        try:
            optimized = prompt_optimizer_agent(prompt, target_agent=service or "general")
            if optimized and len(optimized) > 10:
                active_prompt = optimized
                workflow_log.append({
                    "agent": "prompt-optimizer",
                    "status": "done",
                    "output_preview": optimized[:100]
                })
        except Exception as e:
            workflow_log.append({"agent": "prompt-optimizer", "status": "error", "error": str(e)})

    # ── Step 2: Planning ────────────────────────────────────────────
    plan = None
    if service:
        plan = {
            "agents": [service],
            "reasoning": f"Direct route to {service}",
            "should_evaluate": False
        }
        workflow_log.append({"agent": "planner", "status": "skipped", "reason": "Direct service"})
    elif use_planner:
        try:
            plan = planner_agent(active_prompt, context=context)
            # Enforce max 3 agents in the pipeline
            plan["agents"] = plan["agents"][:3]
            workflow_log.append({
                "agent": "planner",
                "status": "done",
                "plan": plan
            })
        except Exception as e:
            plan = {"agents": ["creative"], "reasoning": "Planner failed", "should_evaluate": False}
            workflow_log.append({"agent": "planner", "status": "error", "error": str(e)})
    else:
        plan = {"agents": ["creative"], "reasoning": "No planner", "should_evaluate": False}

    # ── Step 3: Execute Agent Pipeline (chained) ────────────────────
    # In a multi-agent pipeline, each agent receives:
    #   - The original user prompt
    #   - The accumulated output of all previous agents as context
    # This enables true collaboration: e.g. Research → Creative → Marketing
    
    final_output = ""
    agents_executed = []
    pipeline_context = ""  # Accumulated output from previous agents

    for step_idx, agent_id in enumerate(plan["agents"]):
        executor = AGENT_EXECUTORS.get(agent_id)
        if not executor:
            workflow_log.append({"agent": agent_id, "status": "error", "error": f"Unknown agent"})
            continue

        try:
            import inspect
            sig = inspect.signature(executor)

            # Build enriched context for chained agents
            if step_idx > 0 and pipeline_context:
                chain_context = (
                    f"## Previous Agent Output (Step {step_idx})\n"
                    f"The following was produced by the previous agent in the pipeline. "
                    f"Use it as context and build upon it.\n\n"
                    f"{pipeline_context}\n\n"
                    f"## Original User Request\n{active_prompt}\n\n"
                    f"## Your Task\n"
                    f"You are agent #{step_idx + 1} in a {len(plan['agents'])}-agent pipeline: "
                    f"{' → '.join(plan['agents'])}. "
                    f"Build upon the previous output to fulfill the user's request."
                )
                enriched_prompt = chain_context
            else:
                enriched_prompt = active_prompt

            # Call the agent with context if supported
            if "context" in sig.parameters:
                result = executor(enriched_prompt, context=context)
            else:
                result = executor(enriched_prompt)

            final_output = result

            # Prevent massive base64 image strings from blowing up the context window
            if isinstance(result, str) and result.startswith("data:image/"):
                pipeline_context = "[An image was generated successfully and passed to the user's UI. You cannot see the image itself, but you should acknowledge it was created based on their prompt.]"
                preview = result # Keep the full base64 string for the UI to render
            else:
                pipeline_context = result if isinstance(result, str) else str(result)
                preview = result[:150] if isinstance(result, str) else "Non-text output"

            agents_executed.append(agent_id)
            
            workflow_log.append({
                "agent": agent_id,
                "status": "done",
                "step": step_idx + 1,
                "output_preview": preview
            })
        except Exception as e:
            workflow_log.append({"agent": agent_id, "status": "error", "step": step_idx + 1, "error": str(e)})

    # ── Step 4: Evaluation (optional) ───────────────────────────────
    evaluation = None
    if plan.get("should_evaluate") and final_output and not final_output.startswith("data:image"):
        try:
            evaluation = evaluator_agent(
                prompt, final_output,
                agent_name=" → ".join(agents_executed)
            )
            workflow_log.append({
                "agent": "evaluator",
                "status": "done",
                "verdict": evaluation.get("verdict", "N/A")
            })
            if evaluation.get("verdict") == "FAIL" and evaluation.get("improved_output"):
                final_output = evaluation["improved_output"]
        except Exception as e:
            workflow_log.append({"agent": "evaluator", "status": "error", "error": str(e)})

    # ── Step 5: Store in memory ─────────────────────────────────────
    if final_output:
        agent_label = " → ".join(agents_executed) or "system"
        add_message(session_id, "assistant", final_output, agent_name=agent_label)

    elapsed = round(time.time() - start_time, 2)

    return {
        "orchestrator_status": "success",
        "llm_response": final_output,
        "workflow": workflow_log,
        "plan": plan,
        "evaluation": evaluation,
        "agents_used": agents_executed,
        "pipeline_length": len(agents_executed),
        "session_id": session_id,
        "elapsed_seconds": elapsed
    }
