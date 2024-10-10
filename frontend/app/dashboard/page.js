import { CollapsibleSidebar } from "@/components/CollapsibleSidebar";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Check, Upload, AlertTriangle, Bot } from "lucide-react";

export default function Component() {
  return (
    <div className="flex">
      <CollapsibleSidebar />
      <main className="flex-1 ml-16">
        <div className="flex h-screen">
          <main className="flex-1 p-8 overflow-auto">
            <header className="flex justify-between items-center mb-8">
              <h1 className="text-2xl font-bold">Dashboard</h1>
              {/* <Button variant="ghost">
                <User className="mr-2 h-4 w-4" />
                Profile
              </Button> */}
            </header>
            <div className="grid gap-8 grid-cols-1 lg:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Submit Document</CardTitle>
                  <CardDescription>
                    Upload your document for plagiarism check
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-center w-full">
                    <label
                      htmlFor="dropzone-file"
                      className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100"
                    >
                      <div className="flex flex-col items-center justify-center pt-5 pb-6">
                        <Upload className="w-10 h-10 mb-3 text-gray-400" />
                        <p className="mb-2 text-sm text-gray-500">
                          <span className="font-semibold">Click to upload</span>{" "}
                          or drag and drop
                        </p>
                        <p className="text-xs text-gray-500">
                          PDF or Word Document (MAX. 20MB)
                        </p>
                      </div>
                      <input
                        id="dropzone-file"
                        type="file"
                        className="hidden"
                      />
                    </label>
                  </div>
                  <Button className="w-full mt-4">Submit for Analysis</Button>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Recent Submissions</CardTitle>
                  <CardDescription>
                    Your latest uploaded documents
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-4">
                    <li className="flex justify-between items-center">
                      <span className="text-sm font-medium">
                        Research_Paper_Final.pdf
                      </span>
                      <span className="text-xs text-gray-500">2 hours ago</span>
                    </li>
                    <li className="flex justify-between items-center">
                      <span className="text-sm font-medium">
                        Literature_Review.docx
                      </span>
                      <span className="text-xs text-gray-500">Yesterday</span>
                    </li>
                    <li className="flex justify-between items-center">
                      <span className="text-sm font-medium">
                        Thesis_Chapter_3.pdf
                      </span>
                      <span className="text-xs text-gray-500">3 days ago</span>
                    </li>
                  </ul>
                </CardContent>
              </Card>
              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle>Latest Report: Research_Paper_Final.pdf</CardTitle>
                  <CardDescription>
                    Plagiarism and AI content detection results
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Tabs defaultValue="summary" className="w-full">
                    <TabsList>
                      <TabsTrigger value="summary">Summary</TabsTrigger>
                      <TabsTrigger value="detailed">
                        Detailed Report
                      </TabsTrigger>
                    </TabsList>
                    <TabsContent value="summary" className="mt-6">
                      <div className="grid gap-6 md:grid-cols-2">
                        <div className="space-y-6">
                          <div className="bg-red-50 p-4 rounded-lg">
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center space-x-2">
                                <AlertTriangle className="h-5 w-5 text-red-500" />
                                <span className="font-medium text-sm text-red-700">
                                  Similarity Score
                                </span>
                              </div>
                              <span className="text-2xl font-bold text-red-700">
                                15%
                              </span>
                            </div>
                            <Progress
                              value={15}
                              className="h-2 bg-red-200"
                              indicatorClassName="bg-red-500"
                            />
                          </div>
                          <div className="bg-blue-50 p-4 rounded-lg">
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center space-x-2">
                                <Bot className="h-5 w-5 text-blue-500" />
                                <span className="font-medium text-sm text-blue-700">
                                  AI-Generated Content
                                </span>
                              </div>
                              <span className="text-2xl font-bold text-blue-700">
                                5%
                              </span>
                            </div>
                            <Progress
                              value={5}
                              className="h-2 bg-blue-200"
                              indicatorClassName="bg-blue-500"
                            />
                          </div>
                        </div>
                        <div className="bg-gray-50 p-4 rounded-lg">
                          <h3 className="font-semibold text-lg mb-3">
                            Key Findings
                          </h3>
                          <ul className="space-y-2">
                            {[
                              "15% of content matches other sources",
                              "5% of content likely AI-generated",
                              "3 major sources of similarity identified",
                              "Recommended: Review highlighted sections",
                            ].map((finding, index) => (
                              <li
                                key={index}
                                className="flex items-start space-x-2"
                              >
                                <Check className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
                                <span className="text-sm text-gray-700">
                                  {finding}
                                </span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </TabsContent>
                    <TabsContent value="detailed">
                      <div className="space-y-4">
                        <h3 className="text-lg font-semibold">
                          Similarity Sources
                        </h3>
                        <ul className="space-y-4">
                          <li className="border-b pb-2">
                            <div className="flex justify-between items-start">
                              <div>
                                <p className="font-medium">
                                  Source 1: Academic Journal XYZ
                                </p>
                                <p className="text-sm text-gray-500">
                                  https://journal-xyz.com/article123
                                </p>
                              </div>
                              <span className="text-sm font-medium text-gray-500">
                                7% Match
                              </span>
                            </div>
                          </li>
                          <li className="border-b pb-2">
                            <div className="flex justify-between items-start">
                              <div>
                                <p className="font-medium">
                                  Source 2: Conference Paper ABC
                                </p>
                                <p className="text-sm text-gray-500">
                                  https://conference-abc.org/paper456
                                </p>
                              </div>
                              <span className="text-sm font-medium text-gray-500">
                                5% Match
                              </span>
                            </div>
                          </li>
                          <li className="border-b pb-2">
                            <div className="flex justify-between items-start">
                              <div>
                                <p className="font-medium">
                                  Source 3: Online Article DEF
                                </p>
                                <p className="text-sm text-gray-500">
                                  https://website-def.com/article789
                                </p>
                              </div>
                              <span className="text-sm font-medium text-gray-500">
                                3% Match
                              </span>
                            </div>
                          </li>
                        </ul>
                        <div>
                          <h3 className="text-lg font-semibold mb-2">
                            AI-Generated Content Analysis
                          </h3>
                          <p className="text-sm text-gray-700">
                            Sections with high probability of AI generation are
                            highlighted in the full report. These sections
                            require careful review to ensure academic integrity.
                          </p>
                        </div>
                        <Button className="w-full">Download Full Report</Button>
                      </div>
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>
            </div>
          </main>
        </div>
      </main>
    </div>
  );
}
