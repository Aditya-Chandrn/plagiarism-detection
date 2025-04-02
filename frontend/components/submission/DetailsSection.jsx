"use client";
import React, { useState, useEffect } from "react";
import { useSubmissionContext } from "@/context/SubmissionContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export default function DetailsSection() {
	const { state, dispatch } = useSubmissionContext();
	const [title, setTitle] = useState(state.details.title);
	const [abstract, setAbstract] = useState(state.details.abstract);

	const handleUpdateDetails = () => {
		dispatch({ type: "UPDATE_DETAILS", payload: { title, abstract } });
	};

	return (
		<div className="p-6">
			<h2 className="text-2xl font-semibold mb-4">Manuscript Details</h2>
			<div className="mb-4 gap-5">
				<label className="block mb-1">Manuscript Title:</label>
				<Input
					type="text"
					value={ title }
					onChange={ (e) => setTitle(e.target.value) }
					placeholder="Enter title"
				/>
			</div>
			<div className="mb-2">
				<label className="block mb-1">Abstract (200-300 words):</label>
				<textarea
					value={ abstract }
					onChange={ (e) => setAbstract(e.target.value) }
					className="w-full rounded-md border border-black bg-white px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-black"
					rows={ 6 }
					placeholder="Enter abstract"
				/>
			</div>
			<Button onClick={ handleUpdateDetails }>Save Details</Button>
		</div>
	);
}
