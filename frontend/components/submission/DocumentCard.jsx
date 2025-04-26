"use client";
import React, { useState, useEffect } from "react";
import axios from "axios";
import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function DocumentCard({ doc, onUpdate }) {
	const [submission, setSubmission] = useState(null);
	const [loading, setLoading] = useState(false);
	const [aiOpen, setAiOpen] = useState(false);
	const [simOpen, setSimOpen] = useState(false);
	const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL;

	useEffect(() => {
		axios
			.get(`${BACKEND}/submission/by-document/${doc._id}`, {
				withCredentials: true,
			})
			.then((res) => setSubmission(res.data))
			.catch((err) => console.error("load submission:", err));
	}, [doc._id, BACKEND]);

	const handleRecheck = async (e) => {
		e.stopPropagation();
		setLoading(true);
		try {
			const res = await axios.post(
				`${BACKEND}/document/process/${doc._id}`,
				{},
				{ withCredentials: true }
			);
			onUpdate?.(doc._id, res.data);
		} catch (e) {
			console.error(e);
		} finally {
			setLoading(false);
		}
	};

	const fmt = (n) => (typeof n === "number" ? n.toFixed(2) : n);
	const aiRes = doc.ai_content_result?.length ? doc.ai_content_result : null;
	const avgAi = aiRes
		? aiRes.reduce((sum, r) => sum + r.score, 0) / aiRes.length
		: null;
	const sim = doc.similarity_result?.[0] ?? null;

	if (!submission) {
		return <div className="p-4 mb-4">Loading submission…</div>;
	}

	return (
		<div className="bg-white shadow rounded p-4 mb-4 flex flex-col">
			{/* 1) Header → report */ }
			<Link href={ `/reports/${doc._id}` } className="block">
				<div className="cursor-pointer">
					<h3 className="text-xl font-semibold">{ doc.name }</h3>
					<p className="text-gray-600">
						Uploaded:{ " " }
						{ doc.upload_date
							? new Date(doc.upload_date).toLocaleDateString("en-GB")
							: "N/A" }
					</p>
				</div>
			</Link>

			{/* 2) Submission metadata */ }
			<div className="mt-4 border-t pt-4">
				<h4 className="text-lg font-semibold mb-2">Submission Details</h4>
				<p>
					<span className="font-medium">Title:</span> { submission.title }
				</p>
				<p className="break-words">
					<span className="font-medium">Abstract:</span>{ " " }
					{ submission.abstract }
				</p>
				<p>
					<span className="font-medium">Authors:</span>{ " " }
					{ submission.authors.map((a) => a.name).join(", ") }
				</p>
				<p>
					<span className="font-medium">Keywords:</span>{ " " }
					{ submission.keywords.join(", ") }
				</p>
				<p>
					<span className="font-medium">Reviewers:</span>{ " " }
					{ submission.reviewers.map((r) => r.name).join(", ") }
				</p>
				<p>
					<span className="font-medium">Letter:</span> { submission.letter }
				</p>
			</div>

			{/* 3) AI dropdown */ }
			{ aiRes && (
				<div className="mt-4 p-4 border rounded bg-gray-50">
					<div
						className="flex justify-between items-center cursor-pointer"
						onClick={ (e) => {
							e.stopPropagation();
							setAiOpen((o) => !o);
						} }
					>
						<p className="text-green-700 font-semibold">
							AI Score: { fmt(avgAi) }
						</p>
						<span className="text-xl">{ aiOpen ? "▲" : "▼" }</span>
					</div>
					{ aiOpen && (
						<div className="mt-2">
							{ aiRes.map((r, i) => (
								<p key={ i } className="ml-4 text-green-700">
									{ r.method_name }: { fmt(r.score) }
								</p>
							)) }
						</div>
					) }
				</div>
			) }

			{/* 4) Similarity dropdown */ }
			{ sim && (
				<div className="mt-4 p-4 border rounded bg-gray-50">
					<div
						className="flex justify-between items-center cursor-pointer"
						onClick={ (e) => {
							e.stopPropagation();
							setSimOpen((o) => !o);
						} }
					>
						<p className="text-blue-700 font-semibold">
							Similarity Score: { fmt(sim.score) }
						</p>
						<span className="text-xl">{ simOpen ? "▲" : "▼" }</span>
					</div>
					{ simOpen && (
						<div className="mt-2 ml-4 text-blue-700">
							<p>BERT: { fmt(sim.bert_score) }</p>
							<p>TFIDF: { fmt(sim.tfidf_score) }</p>
						</div>
					) }
				</div>
			) }

			{/* 5) Re-check button */ }
			{ !aiRes && !sim && (
				<div className="mt-4">
					<Button onClick={ handleRecheck } disabled={ loading }>
						{ loading ? "Processing..." : "Find Similarity and AI Score" }
					</Button>
				</div>
			) }
		</div>
	);
}
