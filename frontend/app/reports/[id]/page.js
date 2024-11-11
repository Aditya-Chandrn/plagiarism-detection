import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { AlertTriangle, Bot, ExternalLink } from "lucide-react";

const reportData = {
  documentName: "Research_Paper_Final.pdf",
  content: `Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam euismod, nisi vel consectetur interdum, nisl nunc egestas nunc, vitae tincidunt nisl nunc euismod nunc. Sed euismod, nisi vel consectetur interdum, nisl nunc egestas nunc, vitae tincidunt nisl nunc euismod nunc.

  Nullam euismod, nisi vel consectetur interdum, nisl nunc egestas nunc, vitae tincidunt nisl nunc euismod nunc. Sed euismod, nisi vel consectetur interdum, nisl nunc egestas nunc, vitae tincidunt nisl nunc euismod nunc.
  
  Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam euismod, nisi vel consectetur interdum, nisl nunc egestas nunc, vitae tincidunt nisl nunc euismod nunc. Sed euismod, nisi vel consectetur interdum, nisl nunc egestas nunc, vitae tincidunt nisl nunc euismod nunc.`,
  similarityScore: 15,
  aiGeneratedScore: 5,
  sources: [
    {
      name: "Academic Journal XYZ",
      url: "https://journal-xyz.com/article123",
      match: 7,
    },
    {
      name: "Conference Paper ABC",
      url: "https://conference-abc.org/paper456",
      match: 5,
    },
    {
      name: "Online Article DEF",
      url: "https://website-def.com/article789",
      match: 3,
    },
  ],
};

export default function ReportDetailPage() {
  return (
    <div className="container mx-auto p-4">
      <h1 className="mb-6 text-2xl font-bold">
        Report Detail: {reportData.documentName}
      </h1>
      <div className="grid gap-6 lg:grid-cols-2">
        {/* <Card className="h-[calc(100vh-6rem)] overflow-hidden"> */}
        {/* <CardHeader>
            <CardTitle>Document Content</CardTitle>
          </CardHeader>
          <CardContent> */}
        <embed
          src="http://localhost:8000/document/classification-of-human-and-ai-generated-texts-investigating-1q5bto7ajj.pdf"
          frameborder="0"
          width="100%"
          height="800px"
        ></embed>
        {/* </CardContent>
        </Card> */}

        <Card className="h-[calc(100vh-6rem)] overflow-hidden">
          <CardHeader>
            <CardTitle>Analysis Results</CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[calc(100vh-10rem)] pr-4">
              <div className="space-y-6">
                <div className="space-y-4">
                  <div className="rounded-lg bg-red-50 p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <AlertTriangle className="h-5 w-5 text-red-500" />
                        <span className="font-medium text-sm text-red-700">
                          Similarity Score
                        </span>
                      </div>
                      <span className="text-2xl font-bold text-red-700">
                        {reportData.similarityScore}%
                      </span>
                    </div>
                    <Progress
                      value={reportData.similarityScore}
                      className="h-2 bg-red-200"
                      indicatorClassName="bg-red-500"
                    />
                  </div>
                  <div className="rounded-lg bg-blue-50 p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <Bot className="h-5 w-5 text-blue-500" />
                        <span className="font-medium text-sm text-blue-700">
                          AI-Generated Content
                        </span>
                      </div>
                      <span className="text-2xl font-bold text-blue-700">
                        {reportData.aiGeneratedScore}%
                      </span>
                    </div>
                    <Progress
                      value={reportData.aiGeneratedScore}
                      className="h-2 bg-blue-200"
                      indicatorClassName="bg-blue-500"
                    />
                  </div>
                </div>

                <div>
                  <h3 className="font-semibold text-lg mb-3">
                    Similarity Sources
                  </h3>
                  <ul className="space-y-4">
                    {reportData.sources.map((source, index) => (
                      <li
                        key={index}
                        className="flex items-center justify-between"
                      >
                        <div>
                          <h4 className="font-medium">{source.name}</h4>
                          <a
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-sm text-blue-600 hover:underline flex items-center"
                          >
                            {source.url}{" "}
                            <ExternalLink className="ml-1 h-3 w-3" />
                          </a>
                        </div>
                        <span className="text-sm text-gray-500">
                          {source.match}% Match
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h3 className="font-semibold text-lg mb-3">
                    AI-Generated Content Analysis
                  </h3>
                  <p className="text-sm text-gray-700">
                    Sections with high probability of AI generation are
                    highlighted in the document view. These sections require
                    careful review to ensure academic integrity.
                  </p>
                </div>
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
