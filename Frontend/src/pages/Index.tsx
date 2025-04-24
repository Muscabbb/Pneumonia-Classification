import { useState } from "react";
import { useToast } from "@/components/ui/use-toast";
import Header from "@/components/Header";
import ImageUpload from "@/components/ImageUpload";
import ResultDisplay from "@/components/ResultDisplay";
import HistorySection, { HistoryItem } from "@/components/HistorySection";
import LoadingSpinner from "@/components/LoadingSpinner";
import { analyzeXray } from "@/lib/api";
import { v4 as uuidv4 } from "uuid";

// Type of the response from the server
interface PredictionResponse {
  prediction: string;
  confidence: number;
  filename: string;
}

const Index = () => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [currentPrediction, setCurrentPrediction] =
    useState<PredictionResponse | null>(null);
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const { toast } = useToast();

  const handleSubmit = async (file: File) => {
    setIsLoading(true);
    setCurrentPrediction(null);

    try {
      const result: PredictionResponse = await analyzeXray(file); // ensure the backend matches this structure

      setCurrentPrediction(result);

      const imageUrl = URL.createObjectURL(file);

      const newHistoryItem: HistoryItem = {
        id: uuidv4(),
        imageUrl,
        prediction: result.prediction,
        confidence: result.confidence,
        timestamp: new Date(),
      };

      setHistoryItems((prev) => [newHistoryItem, ...prev]);

      toast({
        title: "Analysis Complete",
        description: `Prediction: ${result.prediction} (${Math.round(
          result.confidence * 100
        )}%)`,
      });
    } catch (error) {
      console.error("Error during analysis:", error);
      toast({
        title: "Analysis Failed",
        description:
          "There was an error analyzing your X-ray image. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />

      <main className="flex-1 container py-6 space-y-8">
        <section className="max-w-3xl mx-auto">
          <h2 className="text-3xl font-bold mb-6 text-center">
            Pneumonia Detection Tool
          </h2>

          <ImageUpload onSubmit={handleSubmit} isLoading={isLoading} />

          {isLoading ? (
            <LoadingSpinner />
          ) : currentPrediction ? (
            <ResultDisplay
              prediction={{
                class: currentPrediction.prediction,
                confidence: currentPrediction.confidence,
              }}
            />
          ) : null}
        </section>

        {historyItems.length > 0 && (
          <section className="max-w-5xl mx-auto mt-10">
            <h3 className="text-xl font-semibold mb-4">History</h3>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <HistorySection historyItems={historyItems} />
            </div>
          </section>
        )}

        <footer className="mt-10 text-center text-sm text-muted-foreground">
          <p>This tool is for educational purposes only.</p>
          <p>
            Always consult with a healthcare professional for proper diagnosis
            and treatment.
          </p>
        </footer>
      </main>
    </div>
  );
};

export default Index;
