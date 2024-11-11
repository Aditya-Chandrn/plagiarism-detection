"use client";
import React, { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { AlertTriangle, Bot, ExternalLink } from "lucide-react";
import { useParams } from "next/navigation";
import axios from "axios";

export default function ReportDetailPage() {
  const [report, setReport] = useState();
  const params = useParams();

  const fetchReport = async () => {
    const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL;

    try {
      const response = await axios.get(`${BACKEND_URL}/document/${params.id}`, {
        withCredentials: true,
      });

      const document = response.data;

      let similarityScore = 0;
      if (document.similarity_result?.length > 0) {
        const totalScore = document.similarity_result.reduce(
          (sum, source) => sum + source.score,
          0
        );

        similarityScore = totalScore / document.similarity_result.length;
      }

      console.log(document);

      similarityScore = Math.ceil(similarityScore * 100);

      const updatedDocument = {
        ...document,
        similarityScore,
      };

      setReport(updatedDocument);
    } catch (error) {
      console.log(error);
    }
  };

  useEffect(() => {
    fetchReport();
  }, []);

  return (
    <div className="container mx-auto p-4">
      <h4 className="mb-6 text-xl font-bold">Report Detail: {report?.name}</h4>
      <div className="grid gap-6 lg:grid-cols-2">
        <embed
          src={`http://localhost:8000/document/file/${report?.name}`}
          frameBorder="0"
          width="100%"
          height="480px"
        ></embed>

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
                        {report?.similarityScore}%
                      </span>
                    </div>
                    <Progress
                      value={report?.similarityScore}
                      className="h-2 bg-red-200"
                      indicatorclassname="bg-red-500"
                    />
                  </div>

                  <div>
                    <h3 className="font-semibold text-lg mb-3">
                      Similarity Sources
                    </h3>
                    <ul className="space-y-4">
                      {report?.similarity_result?.map((result, index) => (
                        <li
                          key={index}
                          className="flex items-center justify-between"
                        >
                          <div>
                            <h4 className="font-medium">
                              {result.source.name}
                            </h4>
                            <a
                              href={result.source.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-sm text-blue-600 hover:underline flex items-center"
                            >
                              {result.source.url}{" "}
                              <ExternalLink className="ml-1 h-3 w-3" />
                            </a>
                          </div>
                          <span className="text-sm text-gray-500">
                            {Math.ceil(result.score * 100)}% Match
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {report?.ai_content_result?.map((res, idx) => {
                    return (
                      <div className="rounded-lg bg-blue-50 p-4" key={idx}>
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            <Bot className="h-5 w-5 text-blue-500" />
                            <span className="font-medium text-sm text-blue-700">
                              AI-Generated Content
                            </span>
                          </div>
                          <span className="text-2xl font-bold text-blue-700">
                            {Math.ceil(res.score * 100)} %
                          </span>
                        </div>
                        <div className="mb-2 text-sm text-blue-600">
                          <strong>Method:</strong> {res.method_name || "N/A"}
                        </div>
                        <Progress
                          value={Math.ceil(res.score * 100)}
                          className="h-2 bg-blue-200"
                          indicatorclassname="bg-blue-500"
                        />
                      </div>
                    );
                  })}
                </div>
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
