import React, { useState, useEffect } from "react";
import axios from "axios";
import Link from "next/link";
import { Card, CardHeader, CardContent, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { FileText, ChevronDown, ChevronUp } from "lucide-react";

export default function DocumentCard({ doc, onUpdate }) {
	const [submission, setSubmission] = useState(null);
	const [loading, setLoading] = useState(false);
	const [expandedLetters, setExpandedLetters] = useState({});
	const [expandedSections, setExpandedSections] = useState({});
	const [showScores, setShowScores] = useState(false);
	const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL;

	useEffect(() => {
		axios
			.get(`${BACKEND}/submission/by-document/${doc._id}`, { withCredentials: true })
			.then((res) => setSubmission(res.data))
			.catch((err) => console.error("load submission:", err));
	}, [doc._id, BACKEND]);

	const aiRes = doc.ai_content_result?.length ? doc.ai_content_result : null;
	const avgAi = aiRes ? aiRes.reduce((sum, r) => sum + r.score, 0) / aiRes.length : 0;
	const sim = doc.similarity_result?.[0] ?? null;

	useEffect(() => {
		if (aiRes || sim) setShowScores(true);
	}, [aiRes, sim]);

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
			setShowScores(true);
		} catch (error) {
			console.error(error);
		} finally {
			setLoading(false);
		}
	};

	const fmt = (num) => (typeof num === "number" ? num.toFixed(2) : num);
	const fmtPercent = (num) => (typeof num === "number" ? (num * 100).toFixed(0) : "N/A");

	const toggleLetter = (key) => {
		setExpandedLetters((prev) => ({ ...prev, [key]: !prev[key] }));
	};

	const toggleSection = (key) => {
		setExpandedSections((prev) => ({ ...prev, [key]: !prev[key] }));
	};

	if (!submission) return <div className="p-4 mb-4">Loading submission…</div>;

	// Letter word count logic
	const letterText = submission.letter || "";
	const letterWordCount = letterText.trim().split(/\s+/).filter((w) => w.length > 0).length;
	const isLetterLong = letterWordCount > 20;

	return (
		<Card className="mb-4">
			<CardHeader className="pb-2">
				<div className="flex justify-between items-start">
					<div className="flex items-center gap-3">
						<FileText className="h-5 w-5 text-gray-500" />
						<div>
							<Link href={ `/reports/${doc._id}` }>
								<CardTitle className="text-lg font-semibold cursor-pointer">
									{ doc.name }
								</CardTitle>
							</Link>
							<p className="text-sm text-gray-500">
								Uploaded: { submission.upload_date ? new Date(submission.upload_date).toLocaleDateString("en-GB") : "N/A" }
							</p>
						</div>
					</div>
					<Badge variant="outline" className="bg-gray-100">
						{ showScores ? "Processed" : "Pending" }
					</Badge>
				</div>
			</CardHeader>

			<CardContent>
				<div className="grid gap-4">
					{/* Submission Details */ }
					<div className="bg-gray-50 p-4 rounded-lg">
						<h3 className="font-medium mb-3">Submission Details</h3>
						<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
							<div>
								<p className="text-sm text-gray-500">Title</p>
								<p className="font-medium">{ submission.title }</p>
							</div>
							<div>
								<p className="text-sm text-gray-500">Abstract</p>
								<p className="font-medium">{ submission.abstract }</p>
							</div>
							<div>
								<p className="text-sm text-gray-500">Authors</p>
								<p className="font-medium">{ submission.authors.map((a) => a.name).join(", ") }</p>
							</div>
							<div>
								<p className="text-sm text-gray-500">Keywords</p>
								<div className="flex flex-wrap gap-1 mt-1">
									{ submission.keywords.map((k) => (
										<Badge key={ k } variant="secondary">{ k }</Badge>
									)) }
								</div>
							</div>
							<div>
								<p className="text-sm text-gray-500">Reviewers</p>
								<p className="font-medium">{ submission.reviewers.map((r) => r.name).join(", ") }</p>
							</div>
							<div>
								<p className="text-sm text-gray-500">Letter</p>
								<div>
									<p className={ `font-medium ${!expandedLetters["letter-1"] && isLetterLong ? "line-clamp-2" : ""}` }>{ letterText }</p>
									{ isLetterLong && (
										<Button
											variant="link"
											className="p-0 h-auto text-sm mt-1"
											onClick={ () => toggleLetter("letter-1") }
										>
											{ expandedLetters["letter-1"] ? "Read less" : "Read more" }
										</Button>
									) }
								</div>
							</div>
						</div>
					</div>

					{/* Action or Scores */ }
					{ !showScores ? (
						<div className="flex justify-center">
							<Button className="px-8" onClick={ handleRecheck } disabled={ loading }>
								{ loading ? "Processing..." : "Find Similarity and AI Score" }
							</Button>
						</div>
					) : (
						<>
							{/* AI Score */ }
							{ aiRes && (
								<div>
									<Button
										variant="ghost"
										className="w-full flex items-center justify-between border border-gray-200 rounded-md px-4 py-2 text-black hover:bg-black hover:text-white focus:ring-0 transition-colors"
										onClick={ () => toggleSection("ai-score") }
									>
										<div className="flex items-center space-x-2">
											<span className="text-base font-semibold">AI Score:</span>
											<span className="text-base font-medium">{ fmtPercent(avgAi) }%</span>
										</div>
										{ expandedSections["ai-score"] ? <ChevronUp /> : <ChevronDown /> }
									</Button>

									{ expandedSections["ai-score"] && (
										<div className="mt-1 px-4 py-2 border border-t-0 border-gray-200 rounded-b-md bg-white space-y-1">
											<p className="text-sm text-black">
												AI detection indicates a { fmtPercent(avgAi) }% chance of AI-generated content.
											</p>
											{ aiRes.map((r, i) => (
												<p key={ i } className="text-sm text-black">
													{ r.method_name }: { fmt(r.score) }
												</p>
											)) }
										</div>
									) }
								</div>
							) }

							{/* Similarity Score */ }
							{ sim && (
								<div>
									<Button
										variant="ghost"
										className="w-full flex items-center justify-between border border-gray-200 rounded-md px-4 py-2 text-black hover:bg-black hover:text-white focus:ring-0 transition-colors"
										onClick={ () => toggleSection("sim-score") }
									>
										<div className="flex items-center space-x-2">
											<span className="text-base font-semibold">Similarity Score:</span>
											<span className="text-base font-medium">{ fmtPercent(sim.score) }%</span>
										</div>
										{ expandedSections["sim-score"] ? <ChevronUp /> : <ChevronDown /> }
									</Button>

									{ expandedSections["sim-score"] && (
										<div className="mt-1 px-4 py-2 border border-t-0 border-gray-200 rounded-b-md bg-white space-y-1">
											<p className="text-sm text-black">
												Overlap of { fmtPercent(sim.score) }% with existing content.
											</p>
											<p className="text-sm text-black">BERT: { fmt(sim.bert_score) }</p>
											<p className="text-sm text-black">TFIDF: { fmt(sim.tfidf_score) }</p>
										</div>
									) }
								</div>
							) }
						</>
					) }
				</div>
			</CardContent>
		</Card>
	);
}