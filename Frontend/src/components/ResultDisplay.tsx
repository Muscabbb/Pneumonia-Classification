import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface ResultDisplayProps {
  prediction: {
    class: string;
    confidence: number;
  } | null;
}

const ResultDisplay = ({ prediction }: ResultDisplayProps) => {
  if (!prediction) return null;

  const isPneumonia = prediction.class === "PNEUMONIA";
  const confidencePercent = Math.round(prediction.confidence * 100);

  return (
    <Card
      className={`mt-6 shadow-card ${
        isPneumonia ? "border-destructive/40" : "border-green-500/40"
      }`}
    >
      <CardHeader className="pb-2">
        <CardTitle className="text-center">
          <span className="text-lg">Diagnosis Result:</span>{" "}
          <span
            className={
              isPneumonia
                ? "text-destructive font-semibold"
                : "text-green-500 font-semibold"
            }
          >
            {prediction.class}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="flex justify-between">
            <span className="text-sm font-medium">Confidence</span>
            <span className="text-sm">{confidencePercent}%</span>
          </div>
          <Progress
            value={confidencePercent}
            className={`${
              isPneumonia ? "bg-destructive/20" : "bg-green-500/20"
            }`}
          />

          <div className="mt-4 pt-2 border-t text-sm text-muted-foreground">
            {isPneumonia ? (
              <p className="text-center">
                The AI model has detected patterns consistent with pneumonia.
                <strong className="block mt-1">
                  Please consult with a healthcare professional for proper
                  diagnosis.
                </strong>
              </p>
            ) : (
              <p className="text-center">
                No patterns consistent with pneumonia were detected.
                <strong className="block mt-1">
                  This is not a medical diagnosis. Always consult with a
                  healthcare professional.
                </strong>
              </p>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ResultDisplay;
