"use client";
import React, { useState } from "react";
import axios from "axios";
import { useSubmissionContext } from "@/context/SubmissionContext";
import { Button } from "@/components/ui/button";
import DocumentsTable from "./DocumentTable";
import { Input } from "../ui/input";

export default function DocumentsSection() {
	const { state, dispatch } = useSubmissionContext();
	const [selectedFile, setSelectedFile] = useState(null);
	const [docType, setDocType] = useState("manuscript");
	const docTypes = [
		"appendix",
		"cover letter",
		"figure",
		"manuscript",
		"table",
		"title page",
		"supplementary material",
		"comment",
	];

	const handleFileChange = (e) => {
		setSelectedFile(e.target.files[0]);
	};

	const addDocument = async () => {
		if (selectedFile) {
			// Create form data for file upload
			const formData = new FormData();
			formData.append("document", selectedFile);
			formData.append("doc_type", docType);

			try {
				const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL;
				// Call the lightweight upload endpoint
				const uploadResponse = await axios.post(
					`${BACKEND_URL}/document/upload/light`,
					formData,
					{
						headers: { "Content-Type": "multipart/form-data" },
						withCredentials: true,
					}
				);
				// Expecting the API to return an object with keys: document_id, name, path, etc.
				const uploadedDoc = {
					id: uploadResponse.data.document_id,
					name: uploadResponse.data.name,
					path: uploadResponse.data.path,
					type: docType,
					// You can add additional properties if needed (e.g. upload date)
				};
				// Dispatch the returned document into your global context
				dispatch({ type: "ADD_DOCUMENT", payload: uploadedDoc });
				setSelectedFile(null);
			} catch (error) {
				console.error("Error uploading document:", error.message);
			}
		}
	};

	return (
		<div className="p-6">
			<h2 className="text-2xl font-semibold mb-4">Documents</h2>
			<div className="flex justify-between mb-4">
				<div className="mb-4">
					<Input
						type="file"
						onChange={ handleFileChange }
						className="p-1"
					/>
				</div>
				<div className="mb-4">
					<label className="mr-2">Document Type:</label>
					<select
						value={ docType }
						onChange={ (e) => setDocType(e.target.value) }
						className="rounded-md border border-black bg-white px-3 py-1 text-sm"
					>
						{ docTypes.map((dt) => (
							<option key={ dt } value={ dt }>
								{ dt }
							</option>
						)) }
					</select>
				</div>
			</div>
			<Button onClick={ addDocument } className="mb-4">
				Add Document
			</Button>
			{ state.documents.length > 0 ? (
				<DocumentsTable />
			) : (
				<p className="text-gray-500">No documents added yet.</p>
			) }
		</div>
	);
}
