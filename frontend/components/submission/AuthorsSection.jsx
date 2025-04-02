"use client";
import React, { useState } from "react";
import { useSubmissionContext } from "@/context/SubmissionContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import AuthorsTable from "./AuthorsTable";

export default function AuthorsSection() {
	const { state, dispatch } = useSubmissionContext();
	const [name, setName] = useState("");
	const [email, setEmail] = useState("");

	const addAuthor = () => {
		if (!name || !email) return;
		const newAuthor = { name, email, type: "Author" };
		dispatch({ type: "ADD_AUTHOR", payload: newAuthor });
		setName("");
		setEmail("");
	};

	return (
		<div className="p-6">
			<h2 className="text-2xl font-semibold mb-4">Authors</h2>
			<div className="flex space-x-2 mb-4 gap-5">
				<Input
					placeholder="Author Name"
					value={ name }
					onChange={ (e) => setName(e.target.value) }
				/>
				<Input
					type="email"
					placeholder="Author Email"
					value={ email }
					onChange={ (e) => setEmail(e.target.value) }
				/>
				<Button onClick={ addAuthor }>Add Author</Button>
			</div>

			{/* Display authors */ }
			{
				state.authors.length > 0 ? (
					<AuthorsTable />
				) : (
					<p className="text-gray-500">No authors added yet.</p>
				)
			}

		</div>
	);
}
