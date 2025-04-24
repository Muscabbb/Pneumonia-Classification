
import { Card, CardContent } from "@/components/ui/card";
import { LoaderCircle } from 'lucide-react';

const LoadingSpinner = () => {
  return (
    <Card className="shadow-card">
      <CardContent className="flex flex-col items-center justify-center py-12">
        <LoaderCircle className="h-12 w-12 text-primary animate-spin" />
        <p className="mt-4 text-muted-foreground">Analyzing X-ray image...</p>
        <p className="text-sm text-muted-foreground mt-1">This may take a few moments</p>
      </CardContent>
    </Card>
  );
};

export default LoadingSpinner;
