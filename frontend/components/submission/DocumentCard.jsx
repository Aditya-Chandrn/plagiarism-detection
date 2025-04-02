"use client";
import React, { useState } from "react";
import axios from "axios";
import { Button } from "@/components/ui/button";
import Link from "next/link";

export default function DocumentCard({ doc, onUpdate }) {
	const [loading, setLoading] = useState(false);
	const [aiDropdownOpen, setAiDropdownOpen] = useState(false);
	const [similarityDropdownOpen, setSimilarityDropdownOpen] = useState(false);

	const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL;

	const handleRecheck = async (e) => {
		// Prevent the card link from triggering when clicking the recheck button.
		e.stopPropagation();
		setLoading(true);
		try {
			const response = await axios.post(
				`${BACKEND_URL}/document/process/${doc._id}`,
				{},
				{ withCredentials: true }
			);
			console.log(response.data);
			if (onUpdate) {
				onUpdate(doc._id, response.data);
			}
		} catch (error) {
			console.error("Error processing document:", error.message);
		} finally {
			setLoading(false);
		}
	};

	const formatScore = (score) =>
		typeof score === "number" ? score.toFixed(2) : score;

	const aiResult =
		doc.ai_content_result && doc.ai_content_result.length > 0
			? doc.ai_content_result
			: null;

	const averageAiScore =
		aiResult &&
		aiResult.reduce((acc, curr) => acc + curr.score, 0) / aiResult.length;

	const similarityResult =
		doc.similarity_result && doc.similarity_result.length > 0
			? doc.similarity_result[0]
			: null;

	return (
		<div className="bg-white shadow rounded p-4 mb-4 flex flex-col">
			{/* Link only wraps the title and metadata */ }
			<Link href={ `/reports/${doc._id}` } className="block">
				<div className="cursor-pointer">
					<h3 className="text-xl font-semibold">{ doc.name }</h3>
					{ doc.upload_date ? (
						<p className="text-gray-600">
							Uploaded: { new Date(doc.upload_date).toLocaleDateString("en-GB") }
						</p>
					) : (
						<p className="text-gray-600">Uploaded: N/A</p>
					) }
				</div>
			</Link>

			{/* AI Score Section (Not Wrapped in Link) */ }
			{ aiResult && (
				<div className="mt-4 p-4 border rounded bg-gray-50">
					<div
						className="flex justify-between items-center cursor-pointer"
						onClick={ (e) => {
							e.stopPropagation(); // Prevents navigation when clicking dropdown
							setAiDropdownOpen(!aiDropdownOpen);
						} }
					>
						<p className="text-green-700 font-semibold">
							AI Score: { formatScore(averageAiScore) }
						</p>
						<span className="text-xl">{ aiDropdownOpen ? "▲" : "▼" }</span>
					</div>
					{ aiDropdownOpen && (
						<div className="mt-2">
							{ aiResult.map((result, index) => (
								<p key={ index } className="text-green-700 ml-4">
									{ result.method_name }: { formatScore(result.score) }
								</p>
							)) }
						</div>
					) }
				</div>
			) }

			{/* Similarity Score Section (Not Wrapped in Link) */ }
			{ similarityResult && (
				<div className="mt-4 p-4 border rounded bg-gray-50">
					<div
						className="flex justify-between items-center cursor-pointer"
						onClick={ (e) => {
							e.stopPropagation(); // Prevents navigation when clicking dropdown
							setSimilarityDropdownOpen(!similarityDropdownOpen);
						} }
					>
						<p className="text-blue-700 font-semibold">
							Similarity Score: { formatScore(similarityResult.score) }
						</p>
						<span className="text-xl">
							{ similarityDropdownOpen ? "▲" : "▼" }
						</span>
					</div>
					{ similarityDropdownOpen && (
						<div className="mt-2 ml-4">
							<p className="text-blue-700">
								BERT Score: { formatScore(similarityResult.bert_score) }
							</p>
							<p className="text-blue-700">
								TFIDF Score: { formatScore(similarityResult.tfidf_score) }
							</p>
						</div>
					) }
				</div>
			) }

			{/* Recheck Button */ }
			{ !(aiResult || similarityResult) && (
				<div className="mt-4">
					<Button onClick={ handleRecheck } disabled={ loading }>
						{ loading ? "Processing..." : "Find Similarity and AI Score" }
					</Button>
				</div>
			) }
		</div>
	);
}
