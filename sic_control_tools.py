from typing import Optional
from langchain_core.tools import tool
from langgraph.config import get_config, get_stream_writer
from src.services.sic_control_service import sic_control_service
from src.state.store import get_store
from src.utils.logger import get_logger

logger = get_logger(__name__)

def _get_thread_context() -> str:
    config = get_config()
    configurable = config.get("configurable", {}) or {}
    return str(configurable.get("thread_id") or "")

def _log_next_action(tool_name: str, result: dict) -> None:
    logger.info("[%s] next_action=%s", tool_name, result.get("next_action", ""))

def _emit_progress(message: str) -> None:
    try:
        writer = get_stream_writer()
        writer({"type": "progress", "content": message})
    except Exception:
        logger.info("[progress_fallback] %s", message)

@tool
async def query_sic_control(
    attribute: str,
    attribute_value: str,
) -> dict:
    """Queries the SIC_CONTROL table for mass component screening rules.
    
    Args:
        attribute: The attribute name (e.g., 'HSA_SERIAL_NUM', 'PCBA_PART_NUM').
        attribute_value: The value of the attribute to screen.
    """
    thread_id = _get_thread_context()
    store = get_store()
    _emit_progress(f"Querying SIC Control for {attribute}={attribute_value}.")
    result = await sic_control_service.query_sic_control(
        attribute=attribute,
        attribute_value=attribute_value,
        thread_id=thread_id,
        store=store,
    )
    _log_next_action("query_sic_control", result)
    return result

@tool
async def add_sic_control_entry(
    attribute: str,
    attribute_value: str,
    operation: str,
    customer: Optional[str] = "ALL",
    descrip: Optional[str] = "Mass screening rule",
) -> dict:
    """Adds a new mass component screening entry to SIC_CONTROL.
    
    Args:
        attribute: The attribute name to block (e.g., 'HSA_SERIAL_NUM').
        attribute_value: The specific value to block.
        operation: The manufacturing stage (e.g., 'VMI', 'CMT').
        customer: The customer ID or 'ALL'.
        descrip: A brief description of why this is being blocked.
    """
    thread_id = _get_thread_context()
    store = get_store()
    _emit_progress(f"Adding SIC Control entry for {attribute}={attribute_value} at {operation}.")
    result = await sic_control_service.add_sic_control_entry(
        attribute=attribute,
        attribute_value=attribute_value,
        operation=operation,
        customer=customer,
        descrip=descrip,
        thread_id=thread_id,
        store=store,
    )
    _log_next_action("add_sic_control_entry", result)
    return result

@tool
async def remove_sic_control_entry(
    attribute: str,
    attribute_value: str,
) -> dict:
    """Removes an existing screening rule from SIC_CONTROL.
    
    Args:
        attribute: The attribute name.
        attribute_value: The attribute value to remove.
    """
    thread_id = _get_thread_context()
    store = get_store()
    _emit_progress(f"Removing SIC Control entry for {attribute}={attribute_value}.")
    result = await sic_control_service.remove_sic_control_entry(
        attribute=attribute,
        attribute_value=attribute_value,
        thread_id=thread_id,
        store=store,
    )
    _log_next_action("remove_sic_control_entry", result)
    return result
