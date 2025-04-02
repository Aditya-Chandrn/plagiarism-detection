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

export default function ReviewerTable() {
	const { state, dispatch } = useSubmissionContext();

	const removeReviewer = (index) => {
		dispatch({ type: "REMOVE_REVIEWER", payload: index });
	};

	return (
		<Table>
			<TableCaption>List of Reviewers</TableCaption>
			<TableHeader>
				<TableRow>
					<TableHead className="w-10">#</TableHead>
					<TableHead className="w-1/3">Name</TableHead>
					<TableHead className="w-1/3">Email</TableHead>
					<TableHead className="w-1/3">Actions</TableHead>
				</TableRow>
			</TableHeader>
			<TableBody>
				{ state.reviewers.map((reviewer, index) => (
					<TableRow key={ index }>
						<TableCell>{ index + 1 }</TableCell>
						<TableCell>{ reviewer.name }</TableCell>
						<TableCell>{ reviewer.email }</TableCell>
						<TableCell>
							<Button
								variant="destructive"
								size="sm"
								className="text-red-500"
								onClick={ () => removeReviewer(index) }
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
