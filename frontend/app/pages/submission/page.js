"use client";
import React, { useState } from "react";
import AuthorsSection from "@/components/submission/AuthorsSection";
import DetailsSection from "@/components/submission/DetailsSection";
import KeywordsSection from "@/components/submission/KeywordsSection";
import DocumentsSection from "@/components/submission/DocumentsSection";
import ReviewersSection from "@/components/submission/ReviewersSection";
import LetterSection from "@/components/submission/LetterSection";
import SubmitSection from "@/components/submission/SubmitSection";

const TABS = [
	{ id: "authors", label: "Authors" },
	{ id: "details", label: "Details" },
	{ id: "keywords", label: "Keywords" },
	{ id: "documents", label: "Documents" },
	{ id: "reviewers", label: "Reviewers" },
	{ id: "letter", label: "Letter" },
	{ id: "submit", label: "Submit" },
];

export default function SubmissionPage() {
	const [activeTab, setActiveTab] = useState("authors");

	const renderTabContent = () => {
		switch (activeTab) {
			case "authors":
				return <AuthorsSection />;
			case "details":
				return <DetailsSection />;
			case "keywords":
				return <KeywordsSection />;
			case "documents":
				return <DocumentsSection />;
			case "reviewers":
				return <ReviewersSection />;
			case "letter":
				return <LetterSection />;
			case "submit":
				return <SubmitSection />;
			default:
				return <AuthorsSection />;
		}
	};

	return (
		<div className="flex flex-col w-full">
			<div className="flex space-x-4 p-4 bg-white justify-center mb-4">
				{ TABS.map((tab) => (
					<button
						key={ tab.id }
						onClick={ () => setActiveTab(tab.id) }
						className={ `px-4 py-2 rounded border transition-colors duration-300 ${activeTab === tab.id
								? "bg-black text-white border-black"
								: "bg-white text-black border-black hover:bg-gray-100"
							}` }
					>
						{ tab.label }
					</button>
				)) }
			</div>
			<div className="flex justify-center px-4 py-4">
				<div className="w-full max-w-5xl">{ renderTabContent() }</div>
			</div>
		</div>
	);
}
