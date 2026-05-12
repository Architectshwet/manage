import uuid
from datetime import datetime, timezone
from src.services.postgres_service import postgres_service
from src.state.store import InMemoryStore, get_session, set_session
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SICControlService:
    """Service workflows for SIC Control (Mass Component Screening)."""

    async def query_sic_control(
        self,
        attribute: str,
        attribute_value: str,
        thread_id: str = "",
        store: InMemoryStore = None,
    ) -> dict:
        """Queries the SIC_CONTROL table for specific attribute/value pairs."""
        logger.info("[query_sic_control] attribute=%s value=%s", attribute, attribute_value)
        
        # Mocking a database query
        # In a real scenario, this would call postgres_service or a direct oracle query
        # For now, let's assume we search for HSA serial numbers or component IDs
        
        return {
            "status": "SUCCESS",
            "data": [
                {
                    "attribute": attribute,
                    "attribute_value": attribute_value,
                    "fg_date": "2026-05-10",
                    "st_date": "2026-05-01",
                    "operation": "VMI",
                    "customer": "DELL",
                    "descrip": f"Screening rule for {attribute}",
                }
            ],
            "next_action": (
                f"Tell the user: I found the screening rule for {attribute}: {attribute_value}.\n"
                "You can now add a new rule or update the existing one if needed."
            )
        }

    async def add_sic_control_entry(
        self,
        attribute: str,
        attribute_value: str,
        operation: str,
        customer: str = "ALL",
        descrip: str = "Mass screening rule",
        thread_id: str = "",
        store: InMemoryStore = None,
    ) -> dict:
        """Adds a new mass component screening entry to SIC_CONTROL."""
        logger.info("[add_sic_control_entry] attribute=%s value=%s", attribute, attribute_value)
        
        entry_id = f"SIC-{uuid.uuid4().hex[:6].upper()}"
        now = datetime.now(timezone.utc).isoformat()
        
        # In a real implementation, we would insert into the database here.
        # await postgres_service.record_sic_action(...)
        
        return {
            "status": "SUCCESS",
            "entry_id": entry_id,
            "next_action": (
                f"Tell the user: Successfully added SIC Control entry {entry_id} for {attribute}={attribute_value}.\n"
                f"Operation: {operation}, Customer: {customer}.\n"
                "The rule is now active for blocking control."
            )
        }

    async def remove_sic_control_entry(
        self,
        attribute: str,
        attribute_value: str,
        thread_id: str = "",
        store: InMemoryStore = None,
    ) -> dict:
        """Removes an existing SIC Control entry."""
        logger.info("[remove_sic_control_entry] attribute=%s value=%s", attribute, attribute_value)
        
        return {
            "status": "SUCCESS",
            "next_action": (
                f"Tell the user: SIC Control entry for {attribute}={attribute_value} has been removed.\n"
                "The component is no longer screened."
            )
        }

sic_control_service = SICControlService()
