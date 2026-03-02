from typing import Optional
from pydantic import BaseModel, Field, validator

class SynthesizeRequest(BaseModel):
    text: str = Field(..., max_length=5000, description="The Arabic text to synthesize. Max 5000 characters.")
    voice_id: str = Field("voice1", description="Persona/Voice ID to use. Default is 'voice1'.")
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Text input cannot be empty or only whitespace.')
        return v

class HealRequest(BaseModel):
    session_id: str = Field(..., description="The session ID returned from /generate.")
    segment_index: int = Field(..., ge=0, description="The array index of the segment to heal.")
    new_text: str = Field(..., max_length=1000, description="The corrected text for this segment.")

    @validator('new_text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Replacement text cannot be empty.')
        return v

class LexiconRequest(BaseModel):
    original: str = Field(..., max_length=100, description="The original word/phrase to catch.")
    corrected: str = Field(..., max_length=100, description="The phonetic spelling/correction to replace it with.")
