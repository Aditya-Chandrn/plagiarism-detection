"use client";
import React from "react";
import { useSubmissionContext } from "@/context/SubmissionContext";
import FinalSubmitComponent from "../ui/final_submit_button";

export default function SubmitSection() {
	const { state } = useSubmissionContext();

	return (
		<div className="p-6 max-w-5xl mx-auto">
			<h2 className="text-3xl font-bold mb-6">Review & Submit</h2>

			{/* Manuscript Details */ }
			<div className="mb-6">
				<h3 className="text-xl font-semibold mb-2">Details</h3>
				<p className="mb-1">
					<span className="font-medium">Title:</span>{ " " }
					<span className="font-normal text-gray-800">{ state.details.title }</span>
				</p>
				<p className="mb-1 break-words">
					<span className="font-medium">Abstract:</span>{ " " }
					<span className="font-normal text-gray-800">{ state.details.abstract }</span>
				</p>
			</div>

			{/* Authors */ }
			<div className="mb-6">
				<h3 className="text-xl font-semibold mb-2">Authors</h3>
				<ul className="list-disc pl-5 text-gray-800">
					{ state.authors.map((author, idx) => (
						<li key={ idx } className="mb-1">
							<span className="font-medium">{ author.name }</span>{ " " }
							(<span className="font-normal">{ author.email }</span>) -{ " " }
							<span className="font-normal">{ author.type }</span>
						</li>
					)) }
				</ul>
			</div>

			{/* Keywords */ }
			<div className="mb-6">
				<h3 className="text-xl font-semibold mb-2">Keywords</h3>
				<ul className="list-disc pl-5 text-gray-800">
					{ state.keywords.map((kw, idx) => (
						<li key={ idx } className="mb-1">{ kw }</li>
					)) }
				</ul>
			</div>

			{/* Documents */ }
			<div className="mb-6">
				<h3 className="text-xl font-semibold mb-2">Documents</h3>
				<ul className="list-disc pl-5 text-gray-800">
					{ state.documents.map((doc, idx) => (
						<li key={ idx } className="mb-1">
							<span className="font-medium">{ doc.name }</span> -{ " " }
							<span className="font-normal">{ doc.type }</span>{ " " }
							{ doc.date && (
								<>
									- <span className="font-normal">{ new Date(doc.date).toLocaleString() }</span>
								</>
							) }
						</li>
					)) }
				</ul>
			</div>

			{/* Reviewers */ }
			<div className="mb-6">
				<h3 className="text-xl font-semibold mb-2">Reviewers</h3>
				<ul className="list-disc pl-5 text-gray-800">
					{ state.reviewers.map((rev, idx) => (
						<li key={ idx } className="mb-1">
							<span className="font-medium">{ rev.name }</span>{ " " }
							(<span className="font-normal">{ rev.email }</span>)
						</li>
					)) }
				</ul>
			</div>

			{/* Letter */ }
			<div className="mb-6">
				<h3 className="text-xl font-semibold mb-2">Letter</h3>
				<p className="text-gray-800">{ state.letter }</p>
			</div>

			{/* Final Submit Button */ }
			<FinalSubmitComponent />
		</div>
	);
}
