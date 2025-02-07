"""Utility & helper functions."""

import base64

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from react_agent.prompts import EXTRACTION_SYSTEM_PROMPT
from react_agent.models import FileContents

def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()

def extract_code(content: str) -> FileContents:
    """Extract JavaScript/TypeScript and CSS code from markdown content using an LLM."""
    model = ChatOpenAI(model="gpt-4o")
    structured_llm = model.with_structured_output(FileContents)

    messages = [
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"Please extract and organize the code from the following text:\n\n{content}"),
    ]
    
    return structured_llm.invoke(messages)

def detect_media_type(base64_data: str) -> str:
    """Detect the media type from base64 data."""
    # Check the first few bytes of the decoded data
    try:
        import magic
        decoded = base64.b64decode(base64_data)
        return magic.from_buffer(decoded, mime=True)
    except ImportError:
        # Fallback: Check common image signatures
        try:
            decoded = base64.b64decode(base64_data[:32])  # First few bytes are enough
            if decoded.startswith(b'\x89PNG\r\n\x1a\n'):
                return 'image/png'
            elif decoded.startswith(b'\xff\xd8'):
                return 'image/jpeg'
            elif decoded.startswith(b'GIF87a') or decoded.startswith(b'GIF89a'):
                return 'image/gif'
            elif decoded.startswith(b'RIFF') and decoded[8:12] == b'WEBP':
                return 'image/webp'
            else:
                return 'image/png'  # Default fallback
        except:
            return 'image/png'  # Default fallback
