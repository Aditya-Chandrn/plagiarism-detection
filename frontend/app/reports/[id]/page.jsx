"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { AlertTriangle, Bot, ExternalLink, ChevronDown } from "lucide-react";
import axios from "axios";

import MarkdownView from "react-showdown";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { SourceSelector } from "@/components/SourceSelector";

const SOURCE_COLORS = [
  "#14b8a6", // teal
  "#8b5cf6", // purple
  "#ef4444", // red
  "#3b82f6", // blue
  "#10b981", // green
  "#f59e0b", // amber
  "#6366f1", // indigo
  "#ec4899", // pink
];

export default function Component() {
  const [report, setReport] = useState(null);
  const [activeSource, setActiveSource] = useState();
  const [fileContent, setFileContent] = useState("");
  const [sources, setSources] = useState([]);
  const params = useParams();

  useEffect(() => {
    const fetchReport = async () => {
      try {
        const response = await axios.get(
          `${process.env.NEXT_PUBLIC_BACKEND_URL}/document/${params.id}`,
          { withCredentials: true }
        );
        const document = response.data;
        const similarityScore = document.similarity_result?.length
          ? Math.ceil(
              (document.similarity_result.reduce(
                (sum, source) => sum + source.score,
                0
              ) /
                document.similarity_result.length) *
                100
            )
          : 0;

        setReport({ ...document, similarityScore });
      } catch (error) {
        console.error(error);
      }
    };

    fetchReport();
  }, [params.id]);

  useEffect(() => {
    if (!report) return;
    const fetchFile = async () => {
      try {
        const filename = report.name.replace(".pdf", ".md");
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_BACKEND_URL}/document/file/${filename}`
        );

        if (!response.ok) {
          throw new Error(`Error: ${response.statusText}`);
        }

        const text = await response.text();
        setFileContent(text);
      } catch (error) {
        console.error("Error fetching file:", error);
      }
    };

    fetchFile();
  }, [report]);

  useEffect(() => {
    if (report?.similarity_result) {
      const dynamicSources = report.similarity_result.map((result, index) => ({
        id: `Source ${index + 1}`,
        name: result.source.name,
        url: result.source.url,
        color: SOURCE_COLORS[index],
        highlights: new Set(result.plagiarized_content.sources),
      }));

      setSources(dynamicSources);
      console.log(dynamicSources);
    }
  }, [report]);

  function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); // Escapes special characters
  }

  const highlightTextInMarkdown = (text) => {
    if (!text) return "";

    let highlightedText = text;
    const sourcesToUse = activeSource
      ? sources.filter((s) => s.id === activeSource)
      : sources;

    sourcesToUse.forEach((source) => {
      const uniqueSentences = new Set(source.highlights);

      uniqueSentences.forEach((phrase) => {
        const escapedPhrase = escapeRegExp(phrase);
        const regex = new RegExp(`(${escapedPhrase})`, "gi");

        highlightedText = highlightedText.replace(
          regex,
          `<span style="background-color: ${source.color}40; color: ${source.color}; font-weight: 500; padding: 1px">$1</span>`
        );
      });
    });

    return highlightedText;
  };

  if (!report) return null;

  return (
    <div className="container mx-auto p-6">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-semibold">
          Report Analysis: {report.name}
        </h1>
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-lg border bg-card shadow-sm">
          <ScrollArea className="h-[calc(100vh-8rem)] w-full rounded-lg p-2">
            <MarkdownView markdown={highlightTextInMarkdown(fileContent)} />
          </ScrollArea>
        </div>

        <Card className="h-[calc(100vh-8rem)]">
          <CardHeader>
            <CardTitle>Analysis Results</CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[calc(100vh-12rem)] pr-4">
              <div className="space-y-6">
                <div className="space-y-4">
                  <h2 className="text-xl font-semibold">Source Highlights</h2>
                  <SourceSelector
                    sources={sources}
                    activeSource={activeSource}
                    onSourceSelect={setActiveSource}
                  />
                </div>

                <Separator />

                <div className="rounded-xl bg-red-50 p-6">
                  <div className="mb-4 flex items-start justify-between">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="h-5 w-5 text-red-600" />
                      <span className="font-medium text-red-900">
                        Similarity Score
                      </span>
                    </div>
                    <span className="text-3xl font-bold text-red-700">
                      {report.similarityScore}%
                    </span>
                  </div>
                  <Progress
                    value={report.similarityScore}
                    className="h-2.5 bg-red-200"
                    indicatorclassname="bg-red-600"
                  />
                </div>

                <Accordion type="single" collapsible className="w-full">
                  <AccordionItem value="similarity-sources">
                    <AccordionTrigger className="text-xl font-semibold">
                      Similarity Sources
                    </AccordionTrigger>
                    <AccordionContent>
                      <div className="space-y-4 pt-4">
                        {report.similarity_result?.map((result, index) => (
                          <div
                            key={index}
                            className="rounded-lg border bg-card p-4 shadow-sm"
                          >
                            <div className="flex items-start justify-between">
                              <div className="space-y-1">
                                <h3 className="font-medium">
                                  {result.source.name}
                                </h3>
                                <a
                                  href={result.source.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="flex items-center text-sm text-blue-600 hover:text-blue-800"
                                >
                                  {result.source.url}
                                  <ExternalLink className="ml-1 h-3 w-3" />
                                </a>
                              </div>
                              <div className="flex flex-col gap-2">
                                <span className="rounded-full bg-gray-100 px-3 py-1 text-sm font-medium">
                                  Bert Score:{" "}
                                  {Math.ceil(result.bert_score * 100)}% Match
                                </span>
                                <span className="rounded-full bg-gray-100 px-3 py-1 text-sm font-medium">
                                  TF-IDF Score:{" "}
                                  {Math.ceil(result.tfidf_score * 100)}% Match
                                </span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>

                <Separator />

                <div className="space-y-4">
                  <h2 className="text-xl font-semibold">
                    AI Detection Results
                  </h2>
                  {report.ai_content_result?.map((result, index) => (
                    <div key={index} className="rounded-xl bg-blue-50 p-6">
                      <div className="mb-4 flex items-start justify-between">
                        <div className="flex items-center gap-2">
                          <Bot className="h-5 w-5 text-blue-600" />
                          <span className="font-medium text-blue-900">
                            AI-Generated Content
                          </span>
                        </div>
                        <span className="text-3xl font-bold text-blue-700">
                          {Math.ceil(result.score * 100)}%
                        </span>
                      </div>
                      <div className="mb-3 text-sm text-blue-700">
                        <span className="font-medium">Method:</span>{" "}
                        {result.method_name || "N/A"}
                      </div>
                      <Progress
                        value={Math.ceil(result.score * 100)}
                        className="h-2.5 bg-blue-200"
                        indicatorclassname="bg-blue-600"
                      />
                    </div>
                  ))}
                </div>
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
