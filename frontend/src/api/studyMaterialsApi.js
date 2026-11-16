import axios from "axios";
import { apiBase, responseDetail } from "./http";

async function postStudyMaterial(baseUrl, path, payload, fallbackError) {
  try {
    const response = await axios.post(`${apiBase(baseUrl)}${path}`, payload, {
      withCredentials: true,
      validateStatus: () => true,
    });

    if (response.status >= 200 && response.status < 300) {
      return { ok: true, data: response.data };
    }

    return { ok: false, error: responseDetail(response) };
  } catch (error) {
    return { ok: false, error: String(error?.message || fallbackError) };
  }
}

export function generateQuiz(baseUrl, { subject, numberQuizzes, numberOptions }) {
  return postStudyMaterial(
    baseUrl,
    "/quizz/",
    {
      subject,
      number_quizzes: numberQuizzes,
      number_options: numberOptions,
    },
    "Could not generate a quiz.",
  );
}

export function generateFlashcards(baseUrl, { topic, numberFlashcards }) {
  return postStudyMaterial(
    baseUrl,
    "/flashcard/",
    {
      topic,
      number_flashcards: numberFlashcards,
    },
    "Could not generate flashcards.",
  );
}
