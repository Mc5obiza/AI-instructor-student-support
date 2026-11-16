import { useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Check,
  ChevronLeft,
  ChevronRight,
  Layers3,
  Loader2,
  RotateCcw,
  Sparkles,
  X,
} from "lucide-react";
import { generateFlashcards, generateQuiz } from "../api/studyMaterialsApi";

function SourceList({ sources }) {
  if (!Array.isArray(sources) || sources.length === 0) {
    return null;
  }

  return (
    <section className="study-sources" aria-label="Course sources">
      <h3>
        <BookOpen size={15} />
        Grounded in your course
      </h3>
      <ul>
        {sources.map((source) => (
          <li key={`${source.source_id}-${source.chunk_id}`}>
            <span className="study-source-number">[{source.source_id}]</span>
            <span>
              <strong>{source.source_file}</strong>
              <span className="study-source-detail">
                {[source.course, source.section].filter(Boolean).join(" · ")}
              </span>
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function StudyForm({
  mode,
  backendUrl,
  onGenerated,
  onClear,
  result,
}) {
  const isQuiz = mode === "quiz";
  const [topic, setTopic] = useState("");
  const [itemCount, setItemCount] = useState(isQuiz ? 5 : 8);
  const [optionCount, setOptionCount] = useState(4);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const label = isQuiz ? "Quiz" : "Flashcards";

  const handleSubmit = async (event) => {
    event.preventDefault();
    const cleanTopic = topic.trim();
    if (!cleanTopic || isLoading) {
      return;
    }

    setError("");
    setIsLoading(true);
    onClear();

    const response = isQuiz
      ? await generateQuiz(backendUrl, {
          subject: cleanTopic,
          numberQuizzes: itemCount,
          numberOptions: optionCount,
        })
      : await generateFlashcards(backendUrl, {
          topic: cleanTopic,
          numberFlashcards: itemCount,
        });

    setIsLoading(false);
    if (!response.ok) {
      setError(response.error || `Could not generate ${label.toLowerCase()}.`);
      return;
    }

    onGenerated(response.data);
  };

  return (
    <form className="study-generator-form" onSubmit={handleSubmit}>
      <label htmlFor={`${mode}-topic`}>
        {isQuiz ? "What would you like to practice?" : "What would you like to review?"}
      </label>
      <div className="study-topic-row">
        <input
          id={`${mode}-topic`}
          type="text"
          value={topic}
          onChange={(event) => setTopic(event.target.value)}
          placeholder="e.g. decision trees, probability, pandas"
          minLength={2}
          maxLength={100}
          required
          disabled={isLoading}
        />
        <button type="submit" className="primary-button study-generate-button" disabled={isLoading}>
          {isLoading ? <Loader2 className="study-spinner" size={16} /> : <Sparkles size={16} />}
          {isLoading ? "Creating…" : `Create ${label.toLowerCase()}`}
        </button>
      </div>
      <div className="study-options-row">
        <label htmlFor={`${mode}-count`}>
          {isQuiz ? "Questions" : "Cards"}
          <select
            id={`${mode}-count`}
            value={itemCount}
            onChange={(event) => setItemCount(Number(event.target.value))}
            disabled={isLoading}
          >
            {(isQuiz ? [3, 5, 7, 10] : [7, 8, 10, 12, 15, 18]).map((count) => (
              <option key={count} value={count}>{count}</option>
            ))}
          </select>
        </label>
        {isQuiz ? (
          <label htmlFor="quiz-options">
            Options per question
            <select
              id="quiz-options"
              value={optionCount}
              onChange={(event) => setOptionCount(Number(event.target.value))}
              disabled={isLoading}
            >
              {[2, 3, 4, 5, 6].map((count) => (
                <option key={count} value={count}>{count}</option>
              ))}
            </select>
          </label>
        ) : null}
        <p>Generated from your indexed course material.</p>
      </div>
      {error ? <p className="study-error" role="alert">{error}</p> : null}
      {isLoading ? (
        <p className="study-loading-note" role="status" aria-live="polite">
          Searching your course material and preparing {label.toLowerCase()}…
        </p>
      ) : null}
      {result ? (
        <button
          type="button"
          className="study-secondary-action"
          onClick={() => {
            setError("");
            onClear();
          }}
        >
          <RotateCcw size={14} />
          Start a new {label.toLowerCase()}
        </button>
      ) : null}
    </form>
  );
}

function QuizPractice({ quiz, onNewQuiz }) {
  const questions = Array.isArray(quiz?.quizz) ? quiz.quizz : [];
  const [questionIndex, setQuestionIndex] = useState(0);
  const [selectedOption, setSelectedOption] = useState(null);
  const currentQuestion = questions[questionIndex];
  const isComplete = questions.length > 0 && questionIndex >= questions.length;
  const score = questions.reduce(
    (total, question, index) => total + (quiz.answers?.[index] === question.correct ? 1 : 0),
    0,
  );

  if (isComplete) {
    return (
      <section className="study-result-card quiz-complete" aria-live="polite">
        <span className="study-result-icon"><Check size={23} /></span>
        <p className="study-card-eyebrow">PRACTICE COMPLETE</p>
        <h2>You scored {score} out of {questions.length}</h2>
        <p>Every question is a chance to strengthen what you know.</p>
        <button type="button" className="primary-button" onClick={onNewQuiz}>
          <RotateCcw size={15} />
          Practice another topic
        </button>
      </section>
    );
  }

  if (!currentQuestion) {
    return null;
  }

  const isCorrect = selectedOption === currentQuestion.correct;
  const handleNext = () => {
    const answers = [...(quiz.answers || [])];
    answers[questionIndex] = selectedOption;
    quiz.onAnswers(answers);
    setQuestionIndex((index) => index + 1);
    setSelectedOption(null);
  };

  return (
    <section className="study-result-card" aria-labelledby="quiz-question-title">
      <div className="study-progress-heading">
        <span>Question {questionIndex + 1} of {questions.length}</span>
        <span>{score} correct</span>
      </div>
      <progress
        className="study-progress"
        max={questions.length}
        value={questionIndex + (selectedOption !== null ? 1 : 0)}
        aria-label={`Question ${questionIndex + 1} of ${questions.length}`}
      />
      <h2 id="quiz-question-title">{currentQuestion.question}</h2>
      <div className="quiz-options" role="group" aria-label="Answer options">
        {currentQuestion.options.map((option, index) => {
          const isSelected = selectedOption === option;
          const isAnswer = selectedOption !== null && option === currentQuestion.correct;
          const stateClass = isAnswer ? "is-correct" : isSelected ? "is-incorrect" : "";

          return (
            <button
              key={`${index}-${option}`}
              type="button"
              className={`quiz-option ${isSelected ? "is-selected" : ""} ${stateClass}`}
              onClick={() => selectedOption === null && setSelectedOption(option)}
              disabled={selectedOption !== null}
              aria-pressed={isSelected}
            >
              <span className="quiz-option-letter">{String.fromCharCode(65 + index)}</span>
              <span>{option}</span>
              {isAnswer ? <Check size={17} /> : null}
              {isSelected && !isCorrect ? <X size={17} /> : null}
            </button>
          );
        })}
      </div>
      {selectedOption !== null ? (
        <div className={`quiz-feedback ${isCorrect ? "is-correct" : "is-incorrect"}`} role="status">
          {isCorrect ? "That’s right." : `Not quite. The correct answer is ${currentQuestion.correct}.`}
        </div>
      ) : (
        <p className="study-answer-hint">Choose an answer to see how you did.</p>
      )}
      <div className="study-card-actions">
        <span>{score} correct so far</span>
        <button
          type="button"
          className="primary-button"
          onClick={handleNext}
          disabled={selectedOption === null}
        >
          {questionIndex === questions.length - 1 ? "See results" : "Next question"}
          <ArrowRight size={15} />
        </button>
      </div>
    </section>
  );
}

function FlashcardPractice({ cards }) {
  const [cardIndex, setCardIndex] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);
  const card = cards?.flashcards?.[cardIndex];

  if (!card) {
    return null;
  }

  const moveCard = (direction) => {
    setCardIndex((index) => (index + direction + cards.flashcards.length) % cards.flashcards.length);
    setIsFlipped(false);
  };

  return (
    <section className="flashcard-practice" aria-label="Flashcard practice">
      <div className="study-progress-heading">
        <span>Card {cardIndex + 1} of {cards.flashcards.length}</span>
        <span>{isFlipped ? "Answer" : "Prompt"}</span>
      </div>
      <progress
        className="study-progress"
        max={cards.flashcards.length}
        value={cardIndex + 1}
        aria-label={`Card ${cardIndex + 1} of ${cards.flashcards.length}`}
      />
      <button
        type="button"
        className={`flashcard-face ${isFlipped ? "is-flipped" : ""}`}
        onClick={() => setIsFlipped((flipped) => !flipped)}
        aria-label={isFlipped ? "Show flashcard question" : "Reveal flashcard answer"}
        aria-live="polite"
      >
        <span className="flashcard-face-label">{isFlipped ? "ANSWER" : "PROMPT"}</span>
        <span className="flashcard-face-text">{isFlipped ? card.back : card.front}</span>
        <span className="flashcard-flip-hint">
          <RotateCcw size={14} />
          Click to {isFlipped ? "see question" : "reveal answer"}
        </span>
      </button>
      <div className="flashcard-controls">
        <button
          type="button"
          className="secondary-button"
          onClick={() => moveCard(-1)}
          disabled={cards.flashcards.length < 2}
          aria-label="Previous flashcard"
        >
          <ChevronLeft size={17} />
          Previous
        </button>
        <button
          type="button"
          className="secondary-button"
          onClick={() => setIsFlipped((flipped) => !flipped)}
        >
          <RotateCcw size={14} />
          Flip card
        </button>
        <button
          type="button"
          className="primary-button"
          onClick={() => moveCard(1)}
          disabled={cards.flashcards.length < 2}
        >
          Next
          <ChevronRight size={17} />
        </button>
      </div>
    </section>
  );
}

export default function StudyToolsPanel({ mode, backendUrl }) {
  const [result, setResult] = useState(null);
  const [answers, setAnswers] = useState([]);
  const isQuiz = mode === "quiz";

  const handleGenerated = (data) => {
    setAnswers([]);
    if (isQuiz) {
      setResult({ ...data, answers: [], onAnswers: setAnswers });
      return;
    }
    setResult(data);
  };

  return (
    <div className="study-workspace">
      <header className="study-heading">
        <div className={`study-heading-icon ${isQuiz ? "quiz-icon" : "flashcard-icon"}`}>
          {isQuiz ? <Sparkles size={19} /> : <Layers3 size={19} />}
        </div>
        <div>
          <p className="study-card-eyebrow">ACTIVE RECALL</p>
          <h1>{isQuiz ? "Quiz yourself" : "Review with flashcards"}</h1>
          <p className="study-heading-copy">
            {isQuiz
              ? "Build confidence with a course-grounded practice quiz."
              : "Turn your course notes into quick, memorable review cards."}
          </p>
        </div>
      </header>

      <StudyForm
        mode={mode}
        backendUrl={backendUrl}
        onGenerated={handleGenerated}
        onClear={() => {
          setResult(null);
          setAnswers([]);
        }}
        result={result}
      />

      {result ? (
        <>
          {isQuiz ? (
            <QuizPractice
              key={result.topic}
              quiz={{ ...result, answers, onAnswers: setAnswers }}
              onNewQuiz={() => {
                setResult(null);
                setAnswers([]);
              }}
            />
          ) : (
            <FlashcardPractice key={result.topic} cards={result} />
          )}
          <SourceList sources={result.sources} />
        </>
      ) : (
        <div className="study-empty-state">
          <span className="study-empty-icon">{isQuiz ? <BookOpen size={20} /> : <Layers3 size={20} />}</span>
          <div>
            <strong>Your {isQuiz ? "quiz" : "flashcards"} will appear here</strong>
            <p>Choose a topic above and we’ll use your indexed course material to build a grounded set.</p>
          </div>
        </div>
      )}
    </div>
  );
}
