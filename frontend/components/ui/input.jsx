import * as React from "react";
import { cn } from "@/lib/utils";

const Input = React.forwardRef(({ className, type, ...props }, ref) => {
  return (
    <input
      type={ type }
      className={ cn(
        "flex h-9 w-full rounded-md border border-black bg-white px-3 py-1 text-sm shadow-sm transition-colors placeholder:text-gray-500 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-black disabled:cursor-not-allowed disabled:opacity-50",
        className
      ) }
      ref={ ref }
      { ...props }
    />
  );
});
Input.displayName = "Input";

export { Input };