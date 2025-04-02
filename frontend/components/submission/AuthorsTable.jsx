"use client";
import React from "react";
import { useSubmissionContext } from "@/context/SubmissionContext";
import {
	Table,
	TableHeader,
	TableBody,
	TableRow,
	TableHead,
	TableCell,
	TableCaption,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";

export default function AuthorsTable() {
	const { state, dispatch } = useSubmissionContext();

	const removeAuthor = (index) => {
		dispatch({ type: "REMOVE_AUTHOR", payload: index });
	};

	return (
		<Table>
			<TableCaption>List of Authors</TableCaption>
			<TableHeader>
				<TableRow>
					<TableHead className="w-10">#</TableHead>
					<TableHead className="w-1/4">Name</TableHead>
					<TableHead className="w-1/4">Email</TableHead>
					<TableHead className="w-1/4">Type</TableHead>
					<TableHead className="w-1/4">Actions</TableHead>
				</TableRow>
			</TableHeader>
			<TableBody>
				{ state.authors.map((author, index) => (
					<TableRow key={ index }>
						<TableCell>{ index + 1 }</TableCell>
						<TableCell>{ author.name }</TableCell>
						<TableCell>{ author.email }</TableCell>
						<TableCell>{ author.type }</TableCell>
						<TableCell>
							<Button
								variant="destructive"
								size="sm"
								className="text-red-500"
								onClick={ () => removeAuthor(index) }
							>
								Remove
							</Button>
						</TableCell>
					</TableRow>
				)) }
			</TableBody>
		</Table>
	);
}
