from __future__ import annotations

import re
from typing import TypeVar

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

try:
	from backend.llm_clients import create_chat_client
	from backend.rag.orchestrator import retrieve_rag_context
	from backend.schemas import (
		Flashcard,
		FlashcardsResponse,
		QuizQuestion,
		QuizResponse,
		SourceAttribution,
	)
except ImportError:
	from llm_clients import create_chat_client
	from rag.orchestrator import retrieve_rag_context
	from schemas import (
		Flashcard,  # pyright: ignore[reportAttributeAccessIssue]
		FlashcardsResponse,  # pyright: ignore[reportAttributeAccessIssue]
		QuizQuestion,  # pyright: ignore[reportAttributeAccessIssue]
		QuizResponse,  # pyright: ignore[reportAttributeAccessIssue]
		SourceAttribution,  # pyright: ignore[reportAttributeAccessIssue]
	)


TOutput = TypeVar("TOutput", bound=BaseModel)


class NoStudyContextError(Exception):
	"""Raised when course retrieval cannot provide supporting context."""


class StudyMaterialGenerationError(Exception):
	"""Raised when the model returns invalid or incomplete study material."""


class _QuizContent(BaseModel):
	quizz: list[QuizQuestion]


class _FlashcardsContent(BaseModel):
	flashcards: list[Flashcard]


def _retrieve_study_context(topic: str, chat_llm: BaseChatModel) -> tuple[str, list[Document]]:
	"""Retrieve quiz/card evidence through the shared RAG orchestration pipeline."""
	result = retrieve_rag_context(question=topic, chat_llm=chat_llm)
	if result.get("status") != "ok":
		raise NoStudyContextError(
			str(result.get("message", "No supporting course material was found for this topic."))
		)

	retrieved = result.get("retrieved", {})
	context = str(retrieved.get("context", "")).strip()
	documents = retrieved.get("ranked_docs", [])
	if not context or not documents:
		raise NoStudyContextError("No supporting course material was found for this topic.")
	return context, documents


def _source_attributions(documents: list[Document]) -> list[SourceAttribution]:
	"""Map retrieved chunk metadata to stable response citations."""
	sources: list[SourceAttribution] = []
	for source_id, document in enumerate(documents, start=1):
		metadata = document.metadata
		values = {
			"course": str(metadata.get("course", "")).strip(),
			"section": str(metadata.get("section", "")).strip(),
			"source_file": str(metadata.get("source_file", "")).strip(),
			"chunk_id": str(metadata.get("chunk_id", "")).strip(),
		}
		if not all(values.values()):
			raise StudyMaterialGenerationError(
				"Retrieved course material is missing source attribution metadata."
			)
		sources.append(SourceAttribution(source_id=source_id, **values))
	return sources


def _validate_citation(text: str, sources: list[SourceAttribution]) -> None:
	"""Ensure generated material cites only sources returned with its response."""
	cited_ids = {int(match) for match in re.findall(r"\[(\d+)\]", text)}
	valid_ids = {source.source_id for source in sources}
	if not cited_ids or not cited_ids.issubset(valid_ids):
		raise StudyMaterialGenerationError(
			"Generated study material must cite the retrieved course sources."
		)


def _generate_structured_output(
	chat_llm: BaseChatModel,
	output_model: type[TOutput],
	system_prompt: str,
	user_prompt: str,
	context: str,
) -> TOutput:
	"""Generate one response constrained to a Pydantic output schema."""
	parser = PydanticOutputParser(pydantic_object=output_model)
	prompt = ChatPromptTemplate.from_messages(
		[
			("system", system_prompt),
			(
				"human",
				"{user_prompt}\n\nRetrieved course context:\n{context}\n\n"
				"Output format instructions:\n{format_instructions}",
			),
		]
	)
	try:
		return (prompt | chat_llm | parser).invoke(
			{
				"user_prompt": user_prompt,
				"context": context,
				"format_instructions": parser.get_format_instructions(),
			}
		)
	except OutputParserException as error:
		raise StudyMaterialGenerationError(
			"The model did not return valid structured study material."
		) from error


def generate_quiz(
	subject: str,
	number_quizzes: int = 3,
	number_options: int = 4,
) -> QuizResponse:
	"""Generate a course-grounded multiple-choice quiz with source citations."""
	topic = subject.strip()
	if not topic:
		raise ValueError("subject is required")
	chat_llm = create_chat_client(temperature=0.2, verbose=False)
	context, documents = _retrieve_study_context(topic, chat_llm)
	sources = _source_attributions(documents)
	output = _generate_structured_output(
		chat_llm=chat_llm,
		output_model=_QuizContent,
		system_prompt=(
			"You create accurate educational multiple-choice questions using only the "
			"retrieved course context. Do not use outside knowledge. Every question and "
			"answer must be answerable from that context. Include the relevant source "
			"citation such as [1] in each question. Each question must have the requested "
			"number of distinct options and exactly one correct option."
		),
		user_prompt=(
			f"Create exactly {number_quizzes} quiz questions about {topic!r}. "
			f"Each question must have exactly {number_options} answer options."
		),
		context=context,
	)
	if len(output.quizz) != number_quizzes:
		raise StudyMaterialGenerationError(
			f"The model returned {len(output.quizz)} questions; {number_quizzes} were requested."
		)
	for question in output.quizz:
		if (
			len(question.options) != number_options
			or len({option.strip().casefold() for option in question.options}) != number_options
			or not all(option.strip() for option in question.options)
			or not any(question.correct.strip() == option.strip() for option in question.options)
		):
			raise StudyMaterialGenerationError(
				"Each quiz question must contain the requested number of distinct options "
				"and identify one of them as correct."
			)
		_validate_citation(question.question, sources)

	return QuizResponse(
		topic=topic,
		nb_questions=number_quizzes,
		quizz=output.quizz,
		sources=sources,
	)


def generate_flashcards(topic: str, number_flashcards: int = 7) -> FlashcardsResponse:
	"""Generate course-grounded flashcards with source citations."""
	clean_topic = topic.strip()
	if not clean_topic:
		raise ValueError("topic is required")
	chat_llm = create_chat_client(temperature=0.2, verbose=False)
	context, documents = _retrieve_study_context(clean_topic, chat_llm)
	sources = _source_attributions(documents)
	output = _generate_structured_output(
		chat_llm=chat_llm,
		output_model=_FlashcardsContent,
		system_prompt=(
			"You create accurate educational flashcards using only the retrieved course "
			"context. Do not use outside knowledge. Each front must ask about a concept "
			"covered by the context, and each back must answer it faithfully. Include a "
			"relevant source citation such as [1] in each back."
		),
		user_prompt=f"Create exactly {number_flashcards} flashcards about {clean_topic!r}.",
		context=context,
	)
	if len(output.flashcards) != number_flashcards:
		raise StudyMaterialGenerationError(
			f"The model returned {len(output.flashcards)} flashcards; "
			f"{number_flashcards} were requested."
		)
	for flashcard in output.flashcards:
		if not flashcard.front.strip() or not flashcard.back.strip():
			raise StudyMaterialGenerationError("Flashcards must have a non-empty front and back.")
		_validate_citation(flashcard.back, sources)

	return FlashcardsResponse(
		topic=clean_topic,
		flashcards=output.flashcards,
		sources=sources,
	)
