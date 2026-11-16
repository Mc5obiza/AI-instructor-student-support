from __future__ import annotations

from pydantic import BaseModel, Field


class QuizRequest(BaseModel):
	subject: str = Field(..., min_length=2, max_length=100)
	number_quizzes: int = Field(default=3, ge=1, le=10)
	number_options: int = Field(default=4, ge=2, le=6)


class FlashcardsRequest(BaseModel):
	topic: str = Field(..., min_length=2, max_length=100)
	number_flashcards: int = Field(default=7, ge=1, le=18)


class QuizQuestion(BaseModel):
	question: str = Field(..., min_length=1)
	options: list[str] = Field(..., min_length=2)
	correct: str = Field(..., min_length=1)


class Flashcard(BaseModel):
	front: str = Field(..., min_length=1)
	back: str = Field(..., min_length=1)


class SourceAttribution(BaseModel):
	source_id: int
	course: str
	section: str
	source_file: str
	chunk_id: str


class QuizResponse(BaseModel):
	topic: str
	nb_questions: int
	quizz: list[QuizQuestion]
	sources: list[SourceAttribution] = Field(default_factory=list)


class FlashcardsResponse(BaseModel):
	topic: str
	flashcards: list[Flashcard]
	sources: list[SourceAttribution] = Field(default_factory=list)
