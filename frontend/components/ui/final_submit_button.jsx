"use client";
import React, { useState } from "react";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { useSubmissionContext } from "@/context/SubmissionContext";
import { useRouter } from "next/navigation";

export default function FinalSubmitComponent({ disabled }) {
	const { state, dispatch } = useSubmissionContext();
	const [loading, setLoading] = useState(false);
	const [message, setMessage] = useState("");
	const router = useRouter();
	const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL;

	const handleFinalSubmit = async () => {
		setLoading(true);
		setMessage("");
		try {
			const submissionPayload = {
				title: state.details.title,
				abstract: state.details.abstract,
				authors: state.authors,
				keywords: state.keywords,
				reviewers: state.reviewers,
				letter: state.letter,
				document_ids: state.documents.map((doc) => doc.id),
			};
			await axios.post(
				`${BACKEND_URL}/submission/submit`,
				submissionPayload,
				{ withCredentials: true }
			);

			setMessage("Submission successful!");
			
			dispatch({ type: "RESET" });
			
			router.push("/pages/submission/summary");
		
		} catch (error) {
			setMessage("Submission failed: " + error.message);
		} finally {
			setLoading(false);
		}
	};

	return (
		<div className="mt-6">
			<Button onClick={ handleFinalSubmit } disabled={ loading || disabled }>
				{ loading ? "Submitting..." : "Final Submit" }
			</Button>
			{ message && <p className="mt-4">{ message }</p> }
		</div>
	);
}
