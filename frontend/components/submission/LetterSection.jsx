"use client";
import React, { useState } from "react";
import { useSubmissionContext } from "@/context/SubmissionContext";
import { Button } from "@/components/ui/button";

export default function LetterSection() {
	const { state, dispatch } = useSubmissionContext();
	const [letter, setLetter] = useState(state.letter);

	const updateLetter = () => {
		dispatch({ type: "UPDATE_LETTER", payload: letter });
	};

	return (
		<div className="p-6">
			<h2 className="text-2xl font-semibold mb-4">Letter</h2>
			<textarea
				value={ letter }
				onChange={ (e) => setLetter(e.target.value) }
				className="w-full rounded-md border border-black bg-white px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-black"
				rows={ 8 }
				placeholder="Enter your letter or additional comments"
			/>
			<Button onClick={ updateLetter } className="mt-2">
				Save Letter
			</Button>
		</div>
	);
}
