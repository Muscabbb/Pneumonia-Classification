import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";

export interface HistoryItem {
  id: string;
  imageUrl: string;
  prediction: string;
  confidence: number;
  timestamp: Date;
}

interface HistorySectionProps {
  historyItems: HistoryItem[];
}

const HistorySection = ({ historyItems }: HistorySectionProps) => {
  if (historyItems.length === 0) {
    return null;
  }

  const formatDate = (date: Date): string => {
    return new Intl.DateTimeFormat("en-US", {
      hour: "numeric",
      minute: "numeric",
      hour12: true,
    }).format(new Date(date));
  };

  return (
    <Card className="shadow-card">
      <CardHeader className="pb-2">
        <CardTitle>Analysis History</CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[250px] pr-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {historyItems.map((item) => (
              <div
                key={item.id}
                className="flex flex-col rounded-md border bg-card overflow-hidden"
              >
                <div className="h-32 overflow-hidden bg-muted">
                  <img
                    src={item.imageUrl}
                    alt="X-ray"
                    className="h-full w-full object-cover"
                  />
                </div>
                <div className="p-3">
                  <div className="flex items-center justify-between">
                    <div
                      className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                        item.prediction === "PNEUMONIA"
                          ? "bg-destructive/10 text-destructive"
                          : "bg-green-500/10 text-green-500"
                      }`}
                    >
                      {item.prediction}
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {formatDate(item.timestamp)}
                    </span>
                  </div>
                  <div className="mt-1 text-xs">
                    <span className="font-medium">Confidence:</span>{" "}
                    {Math.round(item.confidence * 100)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
};

export default HistorySection;
