"use client";
import React, { useState } from "react";
import { useSubmissionContext } from "@/context/SubmissionContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import ReviewerTable from "./ReviewerTable";

export default function ReviewersSection() {
	const { state, dispatch } = useSubmissionContext();
	const [reviewerName, setReviewerName] = useState("");
	const [reviewerEmail, setReviewerEmail] = useState("");

	const addReviewer = () => {
		if (!reviewerName || !reviewerEmail) return;
		const newReviewer = { name: reviewerName, email: reviewerEmail };
		dispatch({ type: "ADD_REVIEWER", payload: newReviewer });
		setReviewerName("");
		setReviewerEmail("");
	};

	return (
		<div className="p-6 ">
			<h2 className="text-2xl font-semibold mb-4">Reviewers</h2>
			<div className="mb-4 flex space-x-2 gap-5">
				<Input
					placeholder="Reviewer Name"
					value={ reviewerName }
					onChange={ (e) => setReviewerName(e.target.value) }
				/>
				<Input
					type="email"
					placeholder="Reviewer Email"
					value={ reviewerEmail }
					onChange={ (e) => setReviewerEmail(e.target.value) }
				/>
				<Button onClick={ addReviewer }>Add Reviewer</Button>
			</div>
			{
				state.reviewers.length > 0 ? (
					<ReviewerTable />
				) : (
					<p className="text-gray-500">No reviewers added yet.</p>
				)
			}
		</div>
	);
}
