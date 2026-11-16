import axios from "axios";
import { apiBase, responseDetail } from "./http";

export async function uploadUserDocument(baseUrl, file, onProgress) {
  const formData = new FormData();
  formData.append("file", file, file.name);

  try {
    const response = await axios.post(`${apiBase(baseUrl)}/documents/upload`, formData, {
      withCredentials: true,
      validateStatus: () => true,
      onUploadProgress: (event) => {
        const total = event.total || file.size;
        if (total > 0 && typeof onProgress === "function") {
          onProgress(Math.min(100, Math.round((event.loaded / total) * 100)));
        }
      },
    });

    if (response.status >= 200 && response.status < 300) {
      return {
        ok: true,
        document: {
          idDocument: String(response.data?.id_document || ""),
          fileName: String(response.data?.file_name || file.name),
          chunkCount: Number(response.data?.chunk_count || 0),
        },
      };
    }

    return { ok: false, error: responseDetail(response) };
  } catch (error) {
    return { ok: false, error: String(error?.message || "Could not upload document") };
  }
}
