import React from "react";
import { TableCell, TableRow } from "@/components/ui/table";

import { AlertTriangle, Bot, Download, Eye } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "./ui/progress";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import Link from "next/link";

const ReportPreview = ({ report }) => {
  return (
    <TableRow key={report.id}>
      <TableCell className="font-medium">{report.title}</TableCell>
      <TableCell>{report.date}</TableCell>
      <TableCell>
        <div className="flex items-center">
          <AlertTriangle className="mr-2 h-4 w-4 text-yellow-500" />
          {report.similarity}%
        </div>
      </TableCell>
      <TableCell>
        <div className="flex items-center">
          <Bot className="mr-2 h-4 w-4 text-blue-500" />
          {report.aiContent}%
        </div>
      </TableCell>
      <TableCell>
        <div className="flex space-x-2">
          {/* <Dialog>
            <DialogTrigger asChild>
             
            </DialogTrigger>
            <DialogContent className="max-w-3xl">
              <DialogHeader>
                <DialogTitle>{report.title} - Report Details</DialogTitle>
                <DialogDescription>
                  Plagiarism and AI content detection results
                </DialogDescription>
              </DialogHeader>
              <div className="grid gap-4">
                <div className="grid gap-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium flex items-center">
                      <AlertTriangle className="mr-2 h-4 w-4 text-yellow-500" />
                      Similarity Score
                    </span>
                    <span className="text-sm font-medium">
                      {report.similarity}%
                    </span>
                  </div>
                  <Progress value={report.similarity} className="w-full" />
                </div>
                <div className="grid gap-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium flex items-center">
                      <Bot className="mr-2 h-4 w-4 text-blue-500" />
                      AI-Generated Content
                    </span>
                    <span className="text-sm font-medium">
                      {report.aiContent}%
                    </span>
                  </div>
                  <Progress value={report.aiContent} className="w-full" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold mb-2">Key Findings</h3>
                  <ul className="list-disc list-inside space-y-1 text-sm">
                    <li>
                      {report.similarity}% of content matches other sources
                    </li>
                    <li>{report.aiContent}% of content likely AI-generated</li>
                    <li>Review highlighted sections in the full report</li>
                  </ul>
                </div>
                <Button className="w-full">
                  <Download className="mr-2 h-4 w-4" />
                  Download Full Report
                </Button>
              </div>
            </DialogContent>
          </Dialog> */}
          <Link href={"/pages/reports/1"}>
            <Button variant="outline" size="sm">
              <Eye className="mr-2 h-4 w-4" />
              View
            </Button>
          </Link>
          <Button variant="outline" size="sm">
            <Download className="mr-2 h-4 w-4" />
            Download
          </Button>
        </div>
      </TableCell>
    </TableRow>
  );
};

export default ReportPreview;
