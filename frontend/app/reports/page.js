"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  Bot,
  Eye,
  Download,
  Search,
} from "lucide-react";
import ReportPreview from "@/components/ReportPreview";
import UploadDocument from "@/components/UploadDocument";

const initialReports = [
  {
    id: 1,
    title: "Research_Paper_Final.pdf",
    date: "2023-05-15",
    similarity: 15,
    aiContent: 5,
  },
  {
    id: 2,
    title: "Literature_Review.docx",
    date: "2023-05-14",
    similarity: 8,
    aiContent: 2,
  },
  {
    id: 3,
    title: "Thesis_Chapter_3.pdf",
    date: "2023-05-12",
    similarity: 12,
    aiContent: 7,
  },
  {
    id: 4,
    title: "Conference_Paper.docx",
    date: "2023-05-10",
    similarity: 20,
    aiContent: 3,
  },
  {
    id: 5,
    title: "Journal_Article_Draft.pdf",
    date: "2023-05-08",
    similarity: 18,
    aiContent: 10,
  },
];

export default function Component() {
  const [reports, setReports] = useState(initialReports);
  const [sortConfig, setSortConfig] = useState({
    key: "date",
    direction: "desc",
  });
  const [searchTerm, setSearchTerm] = useState("");
  const [filterType, setFilterType] = useState("all");

  const sortedReports = [...reports].sort((a, b) => {
    if (a[sortConfig.key] < b[sortConfig.key]) {
      return sortConfig.direction === "asc" ? -1 : 1;
    }
    if (a[sortConfig.key] > b[sortConfig.key]) {
      return sortConfig.direction === "asc" ? 1 : -1;
    }
    return 0;
  });

  const filteredReports = sortedReports.filter(
    (report) =>
      report.title.toLowerCase().includes(searchTerm.toLowerCase()) &&
      (filterType === "all" ||
        (filterType === "high-similarity" && report.similarity > 15) ||
        (filterType === "high-ai" && report.aiContent > 5))
  );

  const handleSort = (key) => {
    setSortConfig((prevConfig) => ({
      key,
      direction:
        prevConfig.key === key && prevConfig.direction === "asc"
          ? "desc"
          : "asc",
    }));
  };

  return (
    <div className="max-w-[1200px] mx-auto px-6 py-8">
      <h1 className="text-3xl font-bold mb-8">Plagiarism Detection Reports</h1>
      <div className="flex justify-between items-center mb-8">
        <div className="relative w-[400px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input
            type="text"
            placeholder="Search reports..."
            className="pl-10 h-11 text-base"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <UploadDocument />
      </div>
      <div className="rounded-lg border">
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead className="w-[300px]">Document Name</TableHead>
              <TableHead>
                <Button
                  variant="ghost"
                  onClick={() => handleSort("date")}
                  className="flex items-center gap-1"
                >
                  Date
                  {sortConfig.key === "date" && (
                    <span>
                      {sortConfig.direction === "asc" ? (
                        <ChevronUp size={16} />
                      ) : (
                        <ChevronDown size={16} />
                      )}
                    </span>
                  )}
                </Button>
              </TableHead>
              <TableHead>
                <Button
                  variant="ghost"
                  onClick={() => handleSort("similarity")}
                  className="flex items-center gap-1"
                >
                  Plagiarism
                  {sortConfig.key === "similarity" && (
                    <span>
                      {sortConfig.direction === "asc" ? (
                        <ChevronUp size={16} />
                      ) : (
                        <ChevronDown size={16} />
                      )}
                    </span>
                  )}
                </Button>
              </TableHead>
              <TableHead>
                <Button
                  variant="ghost"
                  onClick={() => handleSort("ai-content")}
                  className="flex items-center gap-1"
                >
                  AI Content
                  {sortConfig.key === "ai-content" && (
                    <span>
                      {sortConfig.direction === "asc" ? (
                        <ChevronUp size={16} />
                      ) : (
                        <ChevronDown size={16} />
                      )}
                    </span>
                  )}
                </Button>
              </TableHead>
              <TableHead className="text-center">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredReports.map((report) => (
              <TableRow key={report.id}>
                <TableCell className="font-medium">{report.title}</TableCell>
                <TableCell>{report.date}</TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4 text-yellow-500" />
                    <span>{report.similarity}%</span>
                  </div>
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <Bot className="h-4 w-4 text-blue-500" />
                    <span>{report.aiContent}%</span>
                  </div>
                </TableCell>
                <TableCell>
                  <div className="flex justify-center gap-4">
                    <Button variant="ghost" size="sm">
                      <Eye className="h-4 w-4" />
                      <span className="ml-2">View</span>
                    </Button>
                    <Button variant="ghost" size="sm">
                      <Download className="h-4 w-4" />
                      <span className="ml-2">Download</span>
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
