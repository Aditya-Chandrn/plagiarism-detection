"use client";

import * as React from "react";
import { Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

export function SourceSelector({ sources, activeSource, onSourceSelect }) {
  return (
    <TooltipProvider>
      <div className="rounded-lg border bg-card">
        <div className="flex items-center gap-2 border-b p-4">
          <h3 className="text-sm font-medium">Source Highlights</h3>
          <div className="ml-auto">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onSourceSelect(null)}
              className={cn(
                "h-7 text-xs",
                !activeSource && "bg-accent text-accent-foreground"
              )}
            >
              View All
              {!activeSource && <Check className="ml-1 h-3 w-3" />}
            </Button>
          </div>
        </div>
        <ScrollArea>
          <div className="space-y-1 p-2">
            {sources.map((source, idx) => (
              <Button
                key={idx}
                variant="ghost"
                className={cn(
                  "relative w-full justify-start gap-2 pl-8 text-left text-sm",
                  activeSource === source.id &&
                    "bg-accent text-accent-foreground"
                )}
                onClick={() => onSourceSelect(source.id)}
              >
                <div
                  className="absolute left-2 h-3 w-3 rounded-full"
                  style={{ backgroundColor: source.color }}
                />
                <span className="truncate">{source.id}</span>
                <span className="ml-auto flex h-5 min-w-[1.25rem] items-center justify-center rounded-full bg-muted px-1 text-xs tabular-nums">
                  {source.highlights.size}
                </span>
                {activeSource === source.id && (
                  <Check className="ml-1 h-3 w-3" />
                )}
              </Button>
            ))}
          </div>
        </ScrollArea>
      </div>
    </TooltipProvider>
  );
}
