"use client";
import React, { useState } from "react";
import { useSubmissionContext } from "@/context/SubmissionContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { DeleteSvg } from "../ui/delete";

export default function KeywordsSection() {
	const { state, dispatch } = useSubmissionContext();
	const [keyword, setKeyword] = useState("");

	const addKeyword = () => {
		if (keyword.trim()) {
			dispatch({ type: "ADD_KEYWORD", payload: keyword.trim() });
			setKeyword("");
		}
	};

	return (
		<div className="p-6">
			<h2 className="text-2xl font-semibold mb-4">Keywords</h2>
			<div className="mb-4 flex items-center gap-5">
				<Input
					placeholder="Enter a keyword"
					value={ keyword }
					onChange={ (e) => setKeyword(e.target.value) }
				/>
				<Button onClick={ addKeyword } className="ml-2">
					Add
				</Button>
			</div>
			<div className="flex flex-wrap gap-2">
				{ state.keywords.map((kw, idx) => (
					<div
						key={ idx }
						className="bg-black text-white pl-3 py-1 rounded flex items-center"
					>
						{ kw }
						<Button
							variant="destructive"
							onClick={ () =>
								dispatch({ type: "REMOVE_KEYWORD", payload: idx })
							}
							className="bg-black text-white hover:bg-black"
						>
							<DeleteSvg />
						</Button>
					</div>
				)) }
			</div>
		</div>
	);
}
