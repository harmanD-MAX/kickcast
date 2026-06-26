"""
Azure Storage Abstraction
=========================
Thin wrapper around Azure Blob and Table Storage SDKs.

Design choices:
- Containers and tables are auto-created on first use (idempotent).
  This avoids a separate "setup" step and works in both local Azurite
  and production Azure without manual provisioning.
- All operations accept the connection string from Settings, so there's
  zero coupling to a specific environment — swap Azurite for real Azure
  by changing one env var.
- Blob operations work with bytes (for JSON, CSV, model artifacts).
- Table operations work with dicts (Azure Table entities).
"""

import json
import logging
from typing import Any, Optional

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.data.tables import TableServiceClient, TableClient
from azure.storage.blob import BlobServiceClient, ContainerClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Blob Storage
# ---------------------------------------------------------------------------

class BlobStore:
    """Manages Azure Blob Storage containers and blobs."""

    def __init__(self, connection_string: str):
        self._conn_str = connection_string
        self._service = BlobServiceClient.from_connection_string(connection_string)
        self._ensured_containers: set[str] = set()

    def _ensure_container(self, container_name: str) -> ContainerClient:
        """Create the container if it doesn't exist (idempotent)."""
        if container_name not in self._ensured_containers:
            try:
                self._service.create_container(container_name)
                logger.info("Created blob container: %s", container_name)
            except ResourceExistsError:
                pass  # Already exists — fine
            self._ensured_containers.add(container_name)
        return self._service.get_container_client(container_name)

    def upload_blob(
        self, container_name: str, blob_name: str, data: bytes, overwrite: bool = True
    ) -> None:
        """Upload bytes to a blob. Overwrites by default."""
        container = self._ensure_container(container_name)
        container.upload_blob(name=blob_name, data=data, overwrite=overwrite)
        logger.info(
            "Uploaded blob: %s/%s (%d bytes)", container_name, blob_name, len(data)
        )

    def upload_json(
        self, container_name: str, blob_name: str, obj: Any, overwrite: bool = True
    ) -> None:
        """Serialize a Python object to JSON and upload it."""
        data = json.dumps(obj, indent=2, default=str).encode("utf-8")
        self.upload_blob(container_name, blob_name, data, overwrite)

    def download_blob(self, container_name: str, blob_name: str) -> Optional[bytes]:
        """Download a blob's content. Returns None if not found."""
        try:
            container = self._ensure_container(container_name)
            blob_client = container.get_blob_client(blob_name)
            return blob_client.download_blob().readall()
        except ResourceNotFoundError:
            logger.warning("Blob not found: %s/%s", container_name, blob_name)
            return None

    def download_json(self, container_name: str, blob_name: str) -> Optional[Any]:
        """Download a blob and parse it as JSON. Returns None if not found."""
        data = self.download_blob(container_name, blob_name)
        if data is None:
            return None
        return json.loads(data.decode("utf-8"))

    def blob_exists(self, container_name: str, blob_name: str) -> bool:
        """Check whether a blob exists."""
        try:
            container = self._ensure_container(container_name)
            blob_client = container.get_blob_client(blob_name)
            blob_client.get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False


# ---------------------------------------------------------------------------
# Table Storage
# ---------------------------------------------------------------------------

class TableStore:
    """Manages Azure Table Storage tables and entities."""

    def __init__(self, connection_string: str):
        self._conn_str = connection_string
        self._service = TableServiceClient.from_connection_string(connection_string)
        self._ensured_tables: set[str] = set()

    def _ensure_table(self, table_name: str) -> TableClient:
        """Create the table if it doesn't exist (idempotent)."""
        if table_name not in self._ensured_tables:
            try:
                self._service.create_table(table_name)
                logger.info("Created table: %s", table_name)
            except ResourceExistsError:
                pass  # Already exists — fine
            self._ensured_tables.add(table_name)
        return self._service.get_table_client(table_name)

    def upsert_entity(self, table_name: str, entity: dict) -> None:
        """
        Insert or replace an entity in a table.

        The entity dict MUST contain 'PartitionKey' and 'RowKey'.
        All other keys become entity properties.
        """
        table = self._ensure_table(table_name)
        table.upsert_entity(entity=entity)
        logger.debug(
            "Upserted entity: %s [%s / %s]",
            table_name,
            entity.get("PartitionKey"),
            entity.get("RowKey"),
        )

    def query_entities(
        self, table_name: str, filter_str: Optional[str] = None
    ) -> list[dict]:
        """
        Query entities from a table.

        filter_str uses OData syntax, e.g.:
            "PartitionKey eq 'GROUP_A'"
            "status eq 'FINISHED'"
        If None, returns all entities.
        """
        table = self._ensure_table(table_name)
        if filter_str:
            entities = table.query_entities(query_filter=filter_str)
        else:
            entities = table.list_entities()
        return [dict(e) for e in entities]

    def delete_entity(self, table_name: str, partition_key: str, row_key: str) -> None:
        """Delete a single entity."""
        table = self._ensure_table(table_name)
        try:
            table.delete_entity(partition_key=partition_key, row_key=row_key)
            logger.info(
                "Deleted entity: %s [%s / %s]", table_name, partition_key, row_key
            )
        except ResourceNotFoundError:
            logger.warning(
                "Entity not found for deletion: %s [%s / %s]",
                table_name,
                partition_key,
                row_key,
            )

    def count_entities(self, table_name: str) -> int:
        """Count all entities in a table (scans — use sparingly)."""
        table = self._ensure_table(table_name)
        return sum(1 for _ in table.list_entities())
