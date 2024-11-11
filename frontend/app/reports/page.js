"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

import { ChevronDown, ChevronUp, Search } from "lucide-react";
import ReportPreview from "@/components/ReportPreview";

const reports = [
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

export default function ReportsPage() {
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
    <div className="container mx-auto p-6 space-y-8">
      <h1 className="text-3xl font-bold">Plagiarism Detection Reports</h1>
      <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
        <div className="relative w-full sm:w-64">
          <Search
            className="absolute left-2 top-1/2 transform -translate-y-1/2 text-gray-400"
            size={20}
          />
          <Input
            type="text"
            placeholder="Search reports..."
            className="pl-10"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[300px]">
              <Button variant="ghost" onClick={() => handleSort("title")}>
                Document Name
                {sortConfig.key === "title" && (
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
              <Button variant="ghost" onClick={() => handleSort("date")}>
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
              <Button variant="ghost" onClick={() => handleSort("similarity")}>
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
              <Button variant="ghost" onClick={() => handleSort("aiContent")}>
                AI Content
                {sortConfig.key === "aiContent" && (
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
            <TableHead>Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {filteredReports.map((report) => (
            <ReportPreview report={report} />
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
