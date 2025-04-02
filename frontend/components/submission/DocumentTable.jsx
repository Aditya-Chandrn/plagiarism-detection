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

export default function DocumentsTable() {
	const { state, dispatch } = useSubmissionContext();

	const removeDocument = (index) => {
		dispatch({ type: "REMOVE_DOCUMENT", payload: index });
	};

	return (
		<Table>
			<TableCaption>List of Documents</TableCaption>
			<TableHeader>
				<TableRow>
					<TableHead>#</TableHead>
					<TableHead>File Name</TableHead>
					<TableHead>Document Type</TableHead>
					<TableHead>Date</TableHead>
					<TableHead>Actions</TableHead>
				</TableRow>
			</TableHeader>
			<TableBody>
				{ state.documents.map((doc, idx) => (
					<TableRow key={ idx }>
						<TableCell>{ idx + 1 }</TableCell>
						<TableCell>{ doc.name }</TableCell>
						<TableCell>{ doc.type }</TableCell>
						<TableCell>{ new Date(doc.date).toLocaleString() }</TableCell>
						<TableCell>
							<Button
								variant="destructive"
								size="sm"
								onClick={ () => removeDocument(idx) }
								className="text-red-500"
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
