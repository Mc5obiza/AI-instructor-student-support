from __future__ import annotations

import json
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
import pytest

from backend.rag import study_materials

pytestmark = pytest.mark.unit


def _document() -> Document:
	return Document(
		page_content="A decision tree splits data using feature conditions.",
		metadata={
			"course": "Machine Learning",
			"section": "Decision Trees",
			"source_file": "Decision Tree.pdf",
			"chunk_id": "chunk-7",
		},
	)


def test_generate_quiz_uses_retrieved_context_and_returns_sources(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	prompt_texts: list[str] = []
	retrieval_calls: list[dict[str, Any]] = []
	output = {
		"quizz": [
			{
				"question": "What does a decision tree do? [1]",
				"options": ["Splits data", "Sorts files", "Draws charts"],
				"correct": "Splits data",
			}
		]
	}

	def capture_prompt(prompt: Any) -> AIMessage:
		prompt_texts.append(prompt.to_string())
		return AIMessage(content=json.dumps(output))

	monkeypatch.setattr(
		study_materials,
		"retrieve_rag_context",
		lambda **kwargs: (
			retrieval_calls.append(kwargs)
			or {
				"status": "ok",
				"retrieved": {
					"context": "[1] course=Machine Learning | section=Decision Trees | "
					"source=Decision Tree.pdf | chunk_id=chunk-7\nA decision tree splits data.",
					"ranked_docs": [_document()],
				},
			}
		),
	)
	monkeypatch.setattr(
		study_materials,
		"create_chat_client",
		lambda **_: RunnableLambda(capture_prompt),
	)

	result = study_materials.generate_quiz(" decision trees ", 1, 3)

	assert result.topic == "decision trees"
	assert result.nb_questions == 1
	assert result.quizz[0].correct == "Splits data"
	assert result.sources[0].source_file == "Decision Tree.pdf"
	assert "A decision tree splits data." in prompt_texts[0]
	assert "exactly 1 quiz questions" in prompt_texts[0]
	assert retrieval_calls[0]["question"] == "decision trees"
	assert retrieval_calls[0]["chat_llm"] is not None


def test_generate_flashcards_rejects_missing_course_context_before_llm(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(
		study_materials,
		"retrieve_rag_context",
		lambda **_: {"status": "stop", "message": "No matching course chunks."},
	)
	monkeypatch.setattr(
		study_materials,
		"create_chat_client",
		lambda **_: RunnableLambda(
			lambda _: pytest.fail("Generation must not run without supporting context.")
		),
	)

	with pytest.raises(study_materials.NoStudyContextError, match="No matching course chunks"):
		study_materials.generate_flashcards("decision trees", 2)


def test_generate_quiz_rejects_wrong_number_of_generated_questions(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(study_materials, "_retrieve_study_context", lambda *_: (
		"Course context", [_document()]
	))
	monkeypatch.setattr(
		study_materials,
		"create_chat_client",
		lambda **_: RunnableLambda(
			lambda _: AIMessage(
				content=json.dumps(
					{
						"quizz": [
							{
								"question": "Question? [1]",
								"options": ["A", "B"],
								"correct": "A",
							}
						]
					}
				)
			)
		),
	)

	with pytest.raises(study_materials.StudyMaterialGenerationError, match="1 questions"):
		study_materials.generate_quiz("topic", number_quizzes=2, number_options=2)


def test_generate_flashcards_validates_citations_and_returns_sources(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(study_materials, "_retrieve_study_context", lambda *_: (
		"Course context", [_document()]
	))
	monkeypatch.setattr(
		study_materials,
		"create_chat_client",
		lambda **_: RunnableLambda(
			lambda _: AIMessage(
				content=json.dumps(
					{"flashcards": [{"front": "What does a tree do?", "back": "It splits data. [1]"}]}
				)
			)
		),
	)

	result = study_materials.generate_flashcards("decision trees", number_flashcards=1)

	assert result.flashcards[0].back == "It splits data. [1]"
	assert result.sources[0].course == "Machine Learning"


def test_generate_flashcards_rejects_unknown_source_citations(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(study_materials, "_retrieve_study_context", lambda *_: (
		"Course context", [_document()]
	))
	monkeypatch.setattr(
		study_materials,
		"create_chat_client",
		lambda **_: RunnableLambda(
			lambda _: AIMessage(
				content=json.dumps(
					{"flashcards": [{"front": "What does a tree do?", "back": "It splits data. [2]"}]}
				)
			)
		),
	)

	with pytest.raises(study_materials.StudyMaterialGenerationError, match="cite"):
		study_materials.generate_flashcards("decision trees", number_flashcards=1)
