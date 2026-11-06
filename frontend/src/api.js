// All communication with the FastAPI backend lives here.
const BASE = import.meta.env.VITE_API_URL || "";

async function request(path, options) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, options);
  } catch {
    throw new Error("Cannot reach the backend. Start it with: cd backend && uvicorn main:app --port 8000");
  }
  if (!res.ok) {
    let detail = `Server error (${res.status}).`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d) => d.msg).join(" ");
    } catch { /* keep default */ }
    throw new Error(detail);
  }
  return res.json();
}

export async function fetchAlgorithms() {
  return (await request("/api/algorithms")).algorithms;
}

// The same image file and mask PNG are sent once; the backend gives every
// selected algorithm an identical copy.
export function runInpaint(imageFile, maskBlob, keys) {
  const form = new FormData();
  form.append("image", imageFile, imageFile.name);
  form.append("mask", maskBlob, "mask.png");
  form.append("algorithms", keys.join(","));
  return request("/api/inpaint", { method: "POST", body: form });
}
