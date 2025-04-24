// lib/api.ts

export const analyzeXray = async (
  file: File
): Promise<{
  prediction: { class: string; confidence: number; }; class: string; confidence: number 
}> => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("http://localhost:8000/predict", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Prediction failed");
  }

  const data = await response.json();
  console.log(data);
  return data;
};
