"use client";

import { useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { User, LayoutDashboard, FileCheck } from "lucide-react";

const sidebarItems = [
  { icon: LayoutDashboard, label: "Dashboard" },
  { icon: FileCheck, label: "Reports" },
  { icon: User, label: "Profile" },
];

export function CollapsibleSidebar() {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div
      className={cn(
        "fixed left-0 top-0 z-40 flex h-screen flex-col bg-white transition-all duration-300 ease-in-out p-2 border-r border-r-gray-300",
        isExpanded ? "w-56" : "w-20"
      )}
      onMouseEnter={() => setIsExpanded(true)}
      onMouseLeave={() => setIsExpanded(false)}
    >
      <nav className="flex-1">
        <ul className="space-y-1 py-4">
          {sidebarItems.map((item, index) => (
            <li key={index}>
              <Button
                variant="ghost"
                className="w-full h-12 relative flex items-center justify-start px-3"
              >
                <div className="w-10 flex items-center justify-center">
                  <item.icon className="h-5 w-5" />
                </div>
                <span
                  className={cn(
                    "ml-2 absolute left-14 whitespace-nowrap transition-all duration-300",
                    isExpanded ? "opacity-100" : "opacity-0"
                  )}
                >
                  {item.label}
                </span>
              </Button>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  );
}
