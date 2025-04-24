
import { useState, useEffect, useRef } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Upload } from 'lucide-react';
import { useToast } from "@/components/ui/use-toast";

interface ImageUploadProps {
  onSubmit: (file: File) => void;
  isLoading: boolean;
}

const ImageUpload = ({ onSubmit, isLoading }: ImageUploadProps) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();
  
  useEffect(() => {
    if (!selectedFile) {
      setPreview(null);
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) {
      setSelectedFile(null);
      return;
    }

    const file = e.target.files[0];
    
    // Check file type
    if (!file.type.includes('image/jpeg') && !file.type.includes('image/png')) {
      toast({
        title: "Unsupported file format",
        description: "Please upload a JPEG or PNG image.",
        variant: "destructive"
      });
      return;
    }

    // Check file size (limit to 10MB)
    if (file.size > 10 * 1024 * 1024) {
      toast({
        title: "File too large",
        description: "Please upload an image smaller than 10MB.",
        variant: "destructive"
      });
      return;
    }

    setSelectedFile(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      
      if (!file.type.includes('image/jpeg') && !file.type.includes('image/png')) {
        toast({
          title: "Unsupported file format",
          description: "Please upload a JPEG or PNG image.",
          variant: "destructive"
        });
        return;
      }
      
      setSelectedFile(file);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleSubmit = () => {
    if (selectedFile) {
      onSubmit(selectedFile);
    } else {
      toast({
        title: "No image selected",
        description: "Please select an image to analyze.",
        variant: "destructive"
      });
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  return (
    <Card className="shadow-card">
      <CardContent className="pt-6">
        <div 
          className="flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-6 cursor-pointer bg-muted/50 hover:bg-muted transition-colors"
          onClick={triggerFileInput}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          <input 
            ref={fileInputRef}
            type="file" 
            onChange={handleFileChange} 
            className="hidden" 
            accept=".jpg,.jpeg,.png"
          />
          
          {preview ? (
            <div className="w-full max-h-[300px] overflow-hidden rounded-md mb-4">
              <img 
                src={preview} 
                alt="Preview" 
                className="w-full h-auto object-contain" 
              />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-8">
              <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center mb-4">
                <Upload className="h-6 w-6 text-primary" />
              </div>
              <p className="text-sm text-muted-foreground mb-1">Drag and drop your chest X-ray image here</p>
              <p className="text-xs text-muted-foreground">JPG or PNG, max 10MB</p>
            </div>
          )}
        </div>
        
        <div className="flex justify-end mt-4">
          <Button 
            onClick={handleSubmit} 
            disabled={!selectedFile || isLoading}
            className="w-full sm:w-auto"
          >
            {isLoading ? "Analyzing..." : "Analyze X-Ray"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default ImageUpload;
