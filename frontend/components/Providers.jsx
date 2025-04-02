"use client";

import React from "react";
import { SubmissionProvider } from "@/context/SubmissionContext";

export default function Providers({ children }) {
	return <SubmissionProvider>{ children }</SubmissionProvider>;
}