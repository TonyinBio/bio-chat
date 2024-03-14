"""Base schema for data structures."""

import json
import textwrap
import uuid
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from hashlib import sha256
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Sequence

from dataclasses_json import DataClassJsonMixin
from llama_index.core.bridge.pydantic import Field
from llama_index.core.utils import SAMPLE_TEXT, truncate_text
from typing_extensions import Self

DEFAULT_TEXT_NODE_TMPL = "{metadata_str}\n\n{content}"
DEFAULT_METADATA_TMPL = "{key}: {value}"
# NOTE: for pretty printing
TRUNCATE_LENGTH = 350
WRAP_WIDTH = 70


# if TYPE_CHECKING:
#     from haystack.schema import Document as HaystackDocument
#     from llama_index.core.bridge.langchain import Document as LCDocument
#     from semantic_kernel.memory.memory_record import MemoryRecord
    
from llama_index.core.schema import TextNode, BaseNode, MetadataMode


class TripletNode(BaseNode):
    subject: TextNode
    predicate: TextNode
    object: TextNode

    text_template: str = Field(
        default=DEFAULT_TEXT_NODE_TMPL,
        description=(
            "Template for how text is formatted, with {content} and "
            "{metadata_str} placeholders."
        ),
    )
    metadata_template: str = Field(
        default=DEFAULT_METADATA_TMPL,
        description=(
            "Template for how metadata is formatted, with {key} and "
            "{value} placeholders."
        ),
    )
    metadata_seperator: str = Field(
        default="\n",
        description="Separator between metadata fields when converting to string.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "TripletNode"

    @property
    def hash(self) -> str:
        doc_identity = str(self.subject.text) + str(self.predicate.text) + str(self.object.text) + str(self.metadata)
        return str(sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest())

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return 99  # bad

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        """Get object content."""
        metadata_str = self.get_metadata_str(mode=metadata_mode).strip()
        if not metadata_str:
            return str(self.subject.text) + str(self.predicate.text) + str(self.object.text)

        return self.text_template.format(
            content=str(self.subject.text) + str(self.predicate.text) + str(self.object.text), metadata_str=metadata_str
        ).strip()

    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL) -> str:
        """Metadata info string."""
        if mode == MetadataMode.NONE:
            return ""

        usable_metadata_keys = set(self.metadata.keys())
        if mode == MetadataMode.LLM:
            for key in self.excluded_llm_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.remove(key)
        elif mode == MetadataMode.EMBED:
            for key in self.excluded_embed_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.remove(key)

        return self.metadata_seperator.join(
            [
                self.metadata_template.format(key=key, value=str(value))
                for key, value in self.metadata.items()
                if key in usable_metadata_keys
            ]
        )

    def set_content(self, value: Sequence[TextNode]) -> None:
        """Set the content of the node."""
        self.subject = value[0]
        self.predicate = value[1]
        self.object = value[2]

    # def get_node_info(self) -> Dict[str, Any]:
    #     """Get node info."""
    #     return {"start": self.start_char_idx, "end": self.end_char_idx}

    # def get_text(self) -> str:
    #     return self.get_content(metadata_mode=MetadataMode.NONE)

    # @property
    # def node_info(self) -> Dict[str, Any]:
    #     """Deprecated: Get node info."""
    #     return self.get_node_info()


