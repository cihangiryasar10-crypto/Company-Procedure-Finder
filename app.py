from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
from dataclasses import dataclass
from typing import List

import streamlit as st
import whisper
from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


@dataclass
class Chunk:
    source_name: str
    page_number: int
    heading: str
    text: str


def normalize_text(value: str) -> str:
    value = value.replace("\x00", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def is_heading(line: str) -> bool:
    cleaned = line.strip()
    if len(cleaned) < 4 or len(cleaned) > 120:
        return False

    heading_patterns = [
        r"^\d+(\.\d+)*[\)\.]?\s+[A-Z][A-Za-z0-9 ,:/\-]{2,}$",
        r"^[A-Z][A-Z0-9 ,:/\-\(\)]{4,}$",
        r"^(Procedure|Procedures|Steps|Instructions|Method|Operation|Inspection)\b",
    ]
    return any(re.match(pattern, cleaned) for pattern in heading_patterns)


def split_page_into_chunks(source_name: str, page_number: int, page_text: str) -> List[Chunk]:
    text = normalize_text(page_text)
    if not text:
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sections: List[Chunk] = []
    current_heading = f"Page {page_number}"
    buffer: List[str] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        section_text = normalize_text("\n".join(buffer))
        if section_text:
            sections.append(
                Chunk(
                    source_name=source_name,
                    page_number=page_number,
                    heading=current_heading,
                    text=section_text,
                )
            )
        buffer = []

    for line in lines:
        if is_heading(line):
            flush_buffer()
            current_heading = line
            continue
        buffer.append(line)

    flush_buffer()

    expanded_sections: List[Chunk] = []
    for section in sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section.text) if p.strip()]
        if not paragraphs:
            continue

        current_parts: List[str] = []
        current_length = 0
        for paragraph in paragraphs:
            projected = current_length + len(paragraph)
            if current_parts and projected > 1200:
                expanded_sections.append(
                    Chunk(
                        source_name=section.source_name,
                        page_number=section.page_number,
                        heading=section.heading,
                        text="\n\n".join(current_parts),
                    )
                )
                overlap = current_parts[-1:]
                current_parts = overlap + [paragraph]
                current_length = sum(len(item) for item in current_parts)
            else:
                current_parts.append(paragraph)
                current_length = projected

        if current_parts:
            expanded_sections.append(
                Chunk(
                    source_name=section.source_name,
                    page_number=section.page_number,
                    heading=section.heading,
                    text="\n\n".join(current_parts),
                )
            )

    return expanded_sections


def split
