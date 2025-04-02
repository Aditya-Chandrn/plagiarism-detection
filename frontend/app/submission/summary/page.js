"use client";
import React, { useState, useEffect } from "react";
import axios from "axios";
import DocumentCard from "@/components/submission/DocumentCard";
import Link from "next/link";

export default function SubmissionSummary() {
	const [documents, setDocuments] = useState([]);
	const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL;

	// Fetch documents for the current submission (or all submissions)
	useEffect(() => {
		const fetchDocuments = async () => {
			try {
				const response = await axios.get(`${BACKEND_URL}/document/`, { withCredentials: true });
				setDocuments(response.data.documents);
				console.log("Documents fetched:", response.data.documents);

			} catch (error) {
				console.error("Error fetching documents:", error.message);
			}
		};
		fetchDocuments();

	}, [BACKEND_URL]);

	// Update the document in local state when reprocessing completes.
	const handleUpdate = (docId, updatedData) => {
		setDocuments((prevDocs) =>
			prevDocs.map((doc) => (doc.id === docId ? { ...doc, ...updatedData } : doc))
		);
	};

	return (
		<div className="p-6 max-w-5xl mx-auto">
			<h1 className="text-3xl font-bold mb-6">Submission Summary</h1>
			{ documents.length === 0 ? (
				<p>No documents uploaded yet.</p>
			) : (
				documents.map((doc) => (
					<DocumentCard key={ doc.id } doc={ doc } onUpdate={ handleUpdate } />
				))
			) }
		</div>
	);
}
